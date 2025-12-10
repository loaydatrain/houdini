import time
import traceback
import torch
from torch.autograd import Variable
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from HOUDINI.NeuralSynthesizer import NeuralSynthesizer, NeuralSynthesizerResult, _debug_info
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.InterpreterFilters import is_evaluable
from HOUDINI.FnLibraryFunctions import NotHandledException
from HOUDINI.Interpreter.Interpreter import _get_unknown_fns_definitions
from HOUDINI.Synthesizer.MiscUtils import getElapsedTime, formatTime

# EXPERIMENTAL CONFIG
NAS_EXPERIMENTAL_CONFIG = {
    "validity_check": True  # Set to True to enable AZ-NAS validity check (random sampling, no pruning, logging)
}

class NeuralSynthesizerNAS(NeuralSynthesizer):
    """
    A subclass of NeuralSynthesizer that implements AZ-NAS based zero-cost proxy scoring
    and reranking of candidate architectures.
    """

    def init_progs(self):
        """
        Override to do nothing. 
        The base NeuralSynthesizer calls this in __init__, which eagerly consumes the generator.
        For NAS, we want to control the generation loop in solve() to perform reranking.
        """
        pass
    
    def compute_az_nas_score(self, prog, unkSortMap, io_examples) -> Tuple[float, int]:
        """
        Computes a zero-cost proxy score (AZ-NAS style) for a given program without training.
        Combines Expressivity, Trainability, and Complexity.
        Returns: (score, param_count)
        """
        score = 0.0
        param_count = 0
        
        # 1. Determine output type and network definitions
        from HOUDINI.Synthesizer.AST import PPGraphSort

        is_graph = type(self.sort) == PPGraphSort
        try:
            unknown_fns_def = _get_unknown_fns_definitions(unkSortMap, is_graph)
        except Exception as e:
            # If we can't even define the functions, it's a bad candidate.
            return -1.0, 0

        # 2. Create the Neural Modules (PyTorch) - Randomly Initialized
        new_fns_dict, trainable_params = self.interpreter.create_nns(unknown_fns_def)
        
        if not new_fns_dict:
            return 0.0, 0 # No neural components to score, maybe purely functional program?

        # 3. Prepare a single batch of data
        data_loader = self.interpreter._get_data_loader(io_examples)
        if isinstance(data_loader, list): 
            data_loader = data_loader[0]
        
        # Get one batch
        iterator = iter(data_loader)
        try:
            x_np, y_np = next(iterator)
        except StopIteration:
            return 0.0, 0

        # Convert to Torch Variables
        x = Variable(torch.from_numpy(x_np))
        if torch.cuda.is_available(): 
            x = x.cuda()
            for k, v in new_fns_dict.items():
                v.cuda()
        
        # 4. Run Forward Pass & Proxies
        global_vars = {"lib": self.lib}
        global_vars.update(new_fns_dict)
        program_str = repr_py(prog)
        
        try:
            # Hook for Expressivity: Count linear regions
            # We attach hooks to ReLU layers in the new modules
            activation_patterns = []
            
            def hook_fn(module, input, output):
                # output is > 0 for active ReLU
                # We flat the batch dimension
                # input/output shape: [batch, channels, h, w] or [batch, features]
                act = (output > 0).detach().float()
                act = act.view(act.size(0), -1) 
                activation_patterns.append(act.cpu().numpy())

            hooks = []
            for name, module in new_fns_dict.items():
                for layer_name, layer in module.named_modules():
                    if isinstance(layer, torch.nn.ReLU):
                        hooks.append(layer.register_forward_hook(hook_fn))

            # Forward pass
            y_pred = eval(program_str, global_vars)(x)
            
            # --- PROXY 1: EXPRESSIVITY ---
            # Number of unique activation patterns across the batch
            if activation_patterns:
                # Concatenate all layer patterns for each sample: [batch, total_neurons]
                full_pattern = np.concatenate(activation_patterns, axis=1)
                # Count unique rows
                unique_patterns = np.unique(full_pattern, axis=0)
                expressivity_score = unique_patterns.shape[0]
            else:
                expressivity_score = 0
            
            for h in hooks: h.remove()

            # --- PROXY 2: TRAINABILITY (Gradient Norm / Jacobian) ---
            # Simple proxy: Sum of Gradient Norms after backward pass on sum(output)
            # This approximates Jacobian sensitivity
            
            # We need output to be scalar for backward
            if isinstance(y_pred, tuple):
                y_pred = y_pred[1] # Handle (logits, output) tuple if present
                
            # Ensure y_pred is a tensor
            if not isinstance(y_pred, (Variable, torch.Tensor)):
                 trainability_score = 0
            else:
                loss = y_pred.sum()
                self.interpreter.library # Access library if needed
                
                # Clear grads
                for param in trainable_params:
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
                        
                loss.backward()
                
                grad_norm = 0.0
                for param in trainable_params:
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                trainability_score = grad_norm
                
                # Avoid NaN or Inf
                if np.isnan(trainability_score) or np.isinf(trainability_score):
                    trainability_score = 0.0

            # --- PROXY 3: COMPLEXITY ---
            # Parameter count
            param_count = sum(p.numel() for p in trainable_params)
            # We might want to penalize too high complexity, or use it as tie breaker.
            # For AZ-NAS, they often maximize expressivity/trainability while constraining complexity.
            # Here, let's just use log(param_count) as a small additive factor or similar, 
            # or simply return it separately. 
            # For now, let's construct a simple weighted score.
            
            # Normalization (rough heuristics)
            # Expressivity is bounded by batch_size
            norm_expr = expressivity_score / (x.size(0) + 1e-6)
            
            # Trainability can be large; we prefer higher stable gradients but not exploding.
            # Let's use log1p
            norm_train = np.log1p(trainability_score)
            
            # Complexity: we usually want efficient models. 
            # But often larger models perform better. Let's ignore it for ranking unless requested,
            # or treat it as 'larger is slightly better' up to a point.
            
            # Final Score
            # This is a simplified aggregation compared to the full AZ-NAS non-linear aggregation
            score = norm_expr + norm_train 

            return score, param_count
            
        except Exception as e:
            # print(f"Proxy evaluation failed: {e}")
            # traceback.print_exc()
            return -1.0, 0

    def solve(self, io_examples_tr, io_examples_val, io_examples_test) -> List[Tuple[object, float]]:
        """
        Overrides the solve method to implement the harvest-rank-evaluate loop.
        """
        candidates = []
        
        print(f"AZ-NAS: Gathering up to {self.settings.N} candidates...")
        
        # 1. HARVEST PHASE
        n_checked = 0  # Total programs checked
        n_found = 0    # Valid programs found
        
        for prog, unkSortMap in self.synthesizer.genProgs():
             n_checked += 1
             
             # Basic check if runnable
             try:
                 is_ok, ecode = is_evaluable(prog, self)
             except Exception as e:
                 self.log_isevaluable_exception(e, prog, unkSortMap)
                 continue
                 
             if is_ok:
                 # 2. PROXY PHASE
                 nas_score, param_count = self.compute_az_nas_score(prog, unkSortMap, io_examples_tr)
                 candidates.append((nas_score, param_count, prog, unkSortMap))
                 n_found += 1
                 if n_found % 10 == 0:
                     # print('.', end='', flush=True) # This conflicts with the rejection bar
                     pass
             else:
                 if ecode != 2:
                     self.log_rejected_program(prog, ecode)
            
             # Stop if we've checked enough programs OR found enough candidates
             # Note: The original enumerative synthesizer stops if n > N or m >= M.
             # For NAS, we usually want a larger pool (N) to rerank from.
             # If settings.N is 10000, we treat it as the harvest budget.
             if n_checked >= self.settings.N: 
                 break
        
        if self._needs_rejection_newline:
            print()
        
        print(f"\nAZ-NAS: Reranking {len(candidates)} candidates.")
        
        # 3. RERANKING PHASE
        if NAS_EXPERIMENTAL_CONFIG.get("validity_check", False):
            # VALIDITY CHECK MODE: Randomly sample M candidates
            import random
            import os
            
            # Filter out failed proxies if any (-1.0 score)
            valid_candidates = [c for c in candidates if c[0] != -1.0]
            
            if len(valid_candidates) > self.settings.M:
                top_candidates = random.sample(valid_candidates, self.settings.M)
            else:
                top_candidates = valid_candidates
            
            print(f"AZ-NAS Validity Check: Randomly sampled {len(top_candidates)} candidates.")
            
            # Setup logging file
            log_file = "az_nas_validity_data.csv"
            write_header = not os.path.exists(log_file)
            with open(log_file, "a") as f:
                if write_header:
                    f.write("az_nas_score,param_count,accuracy,program\n")
        else:
            # STANDARD MODE: Sort by NAS score (descending)
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Select top M
            top_candidates = candidates[:self.settings.M]
            print(f"AZ-NAS: Selected top {len(top_candidates)} for full evaluation.")

        # Populate the prog_unkinfo_tuples for compatibility if needed by other methods 
        # (like log methods or future progressive tuning)
        self.prog_unkinfo_tuples = [(p, u) for _, _, p, u in top_candidates]

        # 4. EVALUATION PHASE
        top_k_solutions_results = []
        
        for i, (score, param_count, prog, unkSortMap) in enumerate(top_candidates):
            print(f"Eval {i+1}/{len(top_candidates)} | NAS Score: {score:.4f} | Params: {param_count} | Prog: {repr_py(prog)}")
            
            try:
                interpreter_res = self.interpret(prog, unkSortMap, io_examples_tr, io_examples_val, io_examples_test)
                top_k_solutions_results.append((prog, interpreter_res))
                self.update_top_k(top_k_solutions_results)
                
                # Log validity data if enabled
                if NAS_EXPERIMENTAL_CONFIG.get("validity_check", False):
                    # interpret returns dict, usually has 'accuracy' (val)
                    val_acc = interpreter_res.get('accuracy', -1.0)
                    # Escape commas in program string for CSV safety
                    prog_str = repr_py(prog).replace('"', '""')
                    with open("az_nas_validity_data.csv", "a") as f:
                        f.write(f'{score},{param_count},{val_acc},"{prog_str}"\n')

            except NotHandledException:
                self.log_unhandled_program(prog)
            except Exception as e:
                self.log_evaluator_exception(e, prog, unkSortMap)

        # Log evaluated programs for consistency with original class
        for _, _, prog, _ in top_candidates:
             self.evaluated_programs_str.append(repr_py(prog))
             self.evaluated_programs_type_info.append(str(prog))

        self._log_evaluated_programs()
        
        return NeuralSynthesizerResult(top_k_solutions_results)
