import time
import traceback
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import os
import random

from HOUDINI.NeuralSynthesizer import NeuralSynthesizer, NeuralSynthesizerResult, _debug_info, get_lib_names
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.InterpreterFilters import is_evaluable
from HOUDINI.FnLibraryFunctions import NotHandledException
from HOUDINI.Interpreter.Interpreter import _get_unknown_fns_definitions
from HOUDINI.Synthesizer.MiscUtils import getElapsedTime, formatTime

# EXPERIMENTAL CONFIG
NAS_EXPERIMENTAL_CONFIG = {
    "validity_check": False,  # Set to True to enable AZ-NAS validity check (random sampling, no pruning, logging)
    
    # Weights for combining AZ-NAS proxy scores
    # Final score = w_expr * norm_expr + w_prog * norm_prog + w_train * norm_train
    "weight_expressivity": 1.0,
    "weight_progressivity": 1.0,
    "weight_trainability": 1.0,
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
    
    def compute_az_nas_score(self, prog, unkSortMap, io_examples) -> Tuple[float, int, dict]:
        """
        Computes zero-cost proxy scores following AZ-NAS (CVPR 2024).
        
        Metrics (faithful to paper):
        - Expressivity (s^E): Sum of per-block entropy of covariance eigenvalues
        - Progressivity (s^P): Minimum difference between consecutive block expressivities
        - Trainability (s^T): Per-block Jacobian spectral norm penalty, averaged
        - Complexity: Parameter count (paper uses FLOPs)
        
        Returns: (score, param_count, metrics_dict)
        """
        score = 0.0
        param_count = 0
        metrics = {
            "expressivity": 0.0, "progressivity": 0.0, "trainability": 0.0,
            "norm_expr": 0.0, "norm_prog": 0.0, "norm_train": 0.0,
            "num_blocks": 0
        }
        
        from HOUDINI.Synthesizer.AST import PPGraphSort

        is_graph = type(self.sort) == PPGraphSort
        try:
            unknown_fns_def = _get_unknown_fns_definitions(unkSortMap, is_graph)
        except Exception as e:
            return -1.0, 0, metrics

        # Collect library modules (deduplicated)
        lib_module_names = list(dict.fromkeys(get_lib_names(prog)))
        lib_modules = []
        seen_lib_modules = set()
        for name in lib_module_names:
            li = self.lib.get(name)
            if li is None or not isinstance(li.obj, nn.Module):
                continue
            if id(li.obj) in seen_lib_modules:
                continue
            seen_lib_modules.add(id(li.obj))
            lib_modules.append(li.obj)

        # Create new neural modules
        new_fns_dict, trainable_params_new = self.interpreter.create_nns(unknown_fns_def)

        # Combine parameters (deduplicated)
        trainable_params_lib = []
        for m in lib_modules:
            trainable_params_lib.extend(list(m.parameters()))
        all_params = list(trainable_params_new) + trainable_params_lib
        seen_param_ids = set()
        trainable_params = []
        for p in all_params:
            pid = id(p)
            if pid not in seen_param_ids:
                seen_param_ids.add(pid)
                trainable_params.append(p)

        if not trainable_params and not lib_modules and not new_fns_dict:
            return 0.0, 0, metrics

        # Prepare data batch
        data_loader = self.interpreter._get_data_loader(io_examples)
        if isinstance(data_loader, list): 
            data_loader = data_loader[0]
        
        iterator = iter(data_loader)
        try:
            x_np, y_np = next(iterator)
        except StopIteration:
            return 0.0, 0, metrics

        x = Variable(torch.from_numpy(x_np), requires_grad=True)

        use_cuda = torch.cuda.is_available()
        moved_lib_modules = []
        moved_new_modules = []
        if use_cuda:
            x = x.cuda()
            for k, v in new_fns_dict.items():
                v.cuda()
                moved_new_modules.append(v)
            for m in lib_modules:
                m.cuda()
                moved_lib_modules.append(m)
        
        global_vars = {"lib": self.lib}
        global_vars.update(new_fns_dict)
        program_str = repr_py(prog)
        
        try:
            # ================================================================
            # HOOK SETUP: Capture block outputs for AZ-NAS metrics
            # Hook on Linear, Conv2d, LSTM, GRU layers (primary blocks)
            # ================================================================
            block_features = []  # List of {'idx': int, 'input': Tensor, 'output': Tensor}
            block_idx_counter = [0]
            
            def block_hook(module, inp, output):
                """Capture input and output of each primary block"""
                idx = block_idx_counter[0]
                block_idx_counter[0] += 1
                
                # Handle tuple inputs (e.g., LSTM)
                if isinstance(inp, tuple):
                    inp = inp[0]
                if isinstance(output, tuple):
                    output = output[0]
                
                # Store detached copies
                block_features.append({
                    'idx': idx,
                    'input': inp.detach() if isinstance(inp, torch.Tensor) else None,
                    'output': output.detach() if isinstance(output, torch.Tensor) else None,
                    'module': module
                })

            hooks = []
            all_modules = list(new_fns_dict.values()) + lib_modules
            for module in all_modules:
                for layer_name, layer in module.named_modules():
                    # Hook on primary computational blocks
                    if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU, nn.RNN)):
                        hooks.append(layer.register_forward_hook(block_hook))

            # Forward pass
            y_pred = eval(program_str, global_vars)(x)
            
            # Remove hooks
            for h in hooks:
                h.remove()

            num_blocks = len(block_features)
            metrics["num_blocks"] = num_blocks
            
            # ================================================================
            # EXPRESSIVITY (s^E): Per-block entropy of covariance eigenvalues
            # s^E_l = -sum_i (lambda_i * log(lambda_i)) where lambda are normalized eigenvalues
            # s^E = sum_l s^E_l
            # ================================================================
            block_expressivities = []
            
            for bf in block_features:
                feat = bf['output']
                if feat is None:
                    block_expressivities.append(0.0)
                    continue
                
                # Reshape to (C, n) where C = channels, n = batch * spatial
                # For Linear: feat is (B, C) -> transpose to (C, B)
                # For Conv: feat is (B, C, H, W) -> reshape to (C, B*H*W)
                if feat.dim() == 2:
                    # (B, C) -> (C, B)
                    feat_reshaped = feat.t()  # (C, B)
                elif feat.dim() == 3:
                    # (B, C, L) -> (C, B*L)
                    B, C, L = feat.shape
                    feat_reshaped = feat.permute(1, 0, 2).reshape(C, -1)
                elif feat.dim() == 4:
                    # (B, C, H, W) -> (C, B*H*W)
                    B, C, H, W = feat.shape
                    feat_reshaped = feat.permute(1, 0, 2, 3).reshape(C, -1)
                else:
                    block_expressivities.append(0.0)
                    continue
                
                C, n = feat_reshaped.shape
                if n < 2 or C < 1:
                    block_expressivities.append(0.0)
                    continue
                
                try:
                    # Center features
                    feat_centered = feat_reshaped - feat_reshaped.mean(dim=1, keepdim=True)
                    
                    # Covariance matrix V = (1/(n-1)) * f_centered @ f_centered^T
                    # Shape: (C, C)
                    V = (feat_centered @ feat_centered.t()) / (n - 1)
                    
                    # Eigenvalues of covariance
                    eigs = torch.linalg.eigvalsh(V).clamp(min=0)
                    
                    # Normalize to get distribution
                    lam = eigs / (eigs.sum() + 1e-10)
                    
                    # Entropy: s^E_l = -sum(lam * log(lam))
                    sE_l = -(lam * torch.log(lam + 1e-10)).sum().item()
                    block_expressivities.append(sE_l)
                except Exception:
                    block_expressivities.append(0.0)
            
            # Total expressivity: sum of per-block expressivities
            expressivity_score = sum(block_expressivities)
            
            # ================================================================
            # PROGRESSIVITY (s^P): Minimum difference between consecutive blocks
            # s^P = min_{l>=2} (s^E_l - s^E_{l-1})
            # ================================================================
            if len(block_expressivities) >= 2:
                diffs = [block_expressivities[i] - block_expressivities[i-1] 
                         for i in range(1, len(block_expressivities))]
                progressivity_score = min(diffs) if diffs else 0.0
            else:
                progressivity_score = 0.0
            
            # ================================================================
            # TRAINABILITY (s^T): Per-block Jacobian spectral norm penalty
            # c_l = -sigma_l - 1/sigma_l + 2  (equals 0 when sigma_l = 1)
            # s^T = (1/(L-1)) * sum_{l>=2} c_l
            # ================================================================
            if isinstance(y_pred, tuple):
                y_pred = y_pred[1]
            
            block_trainabilities = []
            
            for bf in block_features:
                inp = bf['input']
                out = bf['output']
                
                if inp is None or out is None:
                    continue
                if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
                    continue
                
                try:
                    # Flatten for Jacobian estimation
                    inp_flat = inp.view(inp.size(0), -1)
                    out_flat = out.view(out.size(0), -1)
                    
                    batch_size = out_flat.size(0)
                    out_dim = out_flat.size(1)
                    inp_dim = inp_flat.size(1)
                    
                    # Estimate spectral norm via power iteration with random vectors
                    # Use Hutchinson-style: ||J||_2 ≈ sqrt(E[||Jv||^2]) for unit v
                    num_samples = min(5, out_dim)
                    jvp_norms_sq = []
                    
                    for _ in range(num_samples):
                        # Random unit vector in output space
                        v = torch.randn(batch_size, out_dim, device=out_flat.device)
                        v = v / (v.norm(dim=1, keepdim=True) + 1e-10)
                        
                        # Compute J^T v via backward (gradient of out w.r.t. inp)
                        # We need inp to require grad
                        inp_for_grad = inp.detach().requires_grad_(True)
                        
                        # Re-run the layer
                        module = bf['module']
                        if isinstance(module, nn.Linear):
                            out_recomputed = module(inp_for_grad)
                        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                            out_recomputed = module(inp_for_grad)
                        elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                            out_recomputed, _ = module(inp_for_grad)
                        else:
                            continue
                        
                        out_recomputed_flat = out_recomputed.view(batch_size, -1)
                        
                        # Backward to get J^T v
                        grad_outputs = v[:, :out_recomputed_flat.size(1)]
                        grads = torch.autograd.grad(
                            outputs=out_recomputed_flat,
                            inputs=inp_for_grad,
                            grad_outputs=grad_outputs,
                            retain_graph=False,
                            create_graph=False
                        )[0]
                        
                        # ||J^T v||^2
                        jvp_norm_sq = grads.view(batch_size, -1).norm(dim=1).pow(2).mean().item()
                        jvp_norms_sq.append(jvp_norm_sq)
                    
                    if jvp_norms_sq:
                        # Spectral norm estimate: sqrt(mean(||Jv||^2))
                        sigma_l = np.sqrt(np.mean(jvp_norms_sq) + 1e-10)
                        
                        # AZ-NAS penalty: c_l = -sigma - 1/sigma + 2
                        # This is 0 when sigma=1, negative otherwise
                        c_l = -sigma_l - (1.0 / (sigma_l + 1e-10)) + 2.0
                        block_trainabilities.append(c_l)
                        
                except Exception:
                    pass
            
            # Average trainability across blocks
            if len(block_trainabilities) >= 1:
                trainability_score = np.mean(block_trainabilities)
            else:
                trainability_score = 0.0
            
            # Clamp to avoid extreme values
            trainability_score = max(-10.0, min(2.0, trainability_score))
            
            # ================================================================
            # COMPLEXITY
            # ================================================================
            param_count = sum(p.numel() for p in trainable_params)
            
            # ================================================================
            # NORMALIZATION
            # ================================================================
            # Expressivity: typically 0 to ~50 depending on num blocks and entropy
            # Normalize by num_blocks to make comparable across architectures
            norm_expr = expressivity_score / (num_blocks + 1) if num_blocks > 0 else 0.0
            
            # Progressivity: can be negative (bad) or positive (good)
            # Shift and scale to [0, 1] range: assume range [-5, 5]
            norm_prog = (progressivity_score + 5.0) / 10.0
            norm_prog = max(0.0, min(1.0, norm_prog))
            
            # Trainability: range is roughly [-10, 2], with 0 being optimal
            # Map to [0, 1] where 1 is best (trainability_score = 0)
            # Use 1 - |trainability_score| / 10, clamped
            norm_train = 1.0 - abs(trainability_score) / 10.0
            norm_train = max(0.0, min(1.0, norm_train))
            
            # ================================================================
            # AGGREGATION (weighted sum, configurable)
            # ================================================================
            w_expr = NAS_EXPERIMENTAL_CONFIG.get("weight_expressivity", 1.0)
            w_prog = NAS_EXPERIMENTAL_CONFIG.get("weight_progressivity", 1.0)
            w_train = NAS_EXPERIMENTAL_CONFIG.get("weight_trainability", 1.0)
            score = w_expr * norm_expr + w_prog * norm_prog + w_train * norm_train

            # In the HOUDINI Regime, the az_nas score is inversely correlated with performance
            # So we invert the score
            score = score * -1.0

            metrics = {
                "expressivity": float(expressivity_score),
                "progressivity": float(progressivity_score),
                "trainability": float(trainability_score),
                "norm_expr": float(norm_expr),
                "norm_prog": float(norm_prog),
                "norm_train": float(norm_train),
                "param_count": int(param_count),
                "num_blocks": num_blocks,
                "block_expressivities": [float(x) for x in block_expressivities],
                "weights": {"expr": w_expr, "prog": w_prog, "train": w_train},
            }

            return score, param_count, metrics
            
        except Exception as e:
            # Uncomment for debugging:
            # print(f"Proxy evaluation failed: {e}")
            # traceback.print_exc()

            return -42.0, 0, metrics
        finally:
            if use_cuda:
                for m in moved_new_modules:
                    m.cpu()
                for m in moved_lib_modules:
                    m.cpu()

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
                 nas_score, param_count, nas_metrics = self.compute_az_nas_score(prog, unkSortMap, io_examples_tr)
                 candidates.append((nas_score, param_count, nas_metrics, prog, unkSortMap))
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
            # Setup logging file (create with header if missing; do NOT delete per task)
            log_file = "az_nas_validity_data.csv"
            if not os.path.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("az_nas_score,param_count,expressivity,progressivity,trainability,norm_expr,norm_prog,norm_train,accuracy,program\n")

            # VALIDITY CHECK MODE: Randomly sample M candidates
            # Filter out failed proxies if any (-1.0 score)
            valid_candidates = [c for c in candidates if c[0] != -1.0]
            
            if len(valid_candidates) > self.settings.M:
                top_candidates = random.sample(valid_candidates, self.settings.M)
            else:
                top_candidates = valid_candidates
            
            print(f"AZ-NAS Validity Check: Randomly sampled {len(top_candidates)} candidates.")
        else:
            # STANDARD MODE: Sort by NAS score (descending)
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Select top half of candidates with a minimum of 1 candidate
            top_candidates = candidates[:max(1, len(candidates)//2)]
            print(f"AZ-NAS: Selected top {len(top_candidates)} for full evaluation.")

        # Populate the prog_unkinfo_tuples for compatibility if needed by other methods 
        # (like log methods or future progressive tuning)
        self.prog_unkinfo_tuples = [(p, u) for _, _, _, p, u in top_candidates]

        # 4. EVALUATION PHASE
        top_k_solutions_results = []
        
        for i, (score, param_count, nas_metrics, prog, unkSortMap) in enumerate(top_candidates):
            print(f"Eval {i+1}/{len(top_candidates)} | NAS: {score:.3f} | "
                  f"E:{nas_metrics.get('norm_expr', 0):.2f} P:{nas_metrics.get('norm_prog', 0):.2f} T:{nas_metrics.get('norm_train', 0):.2f} | "
                  f"Params: {param_count} | {repr_py(prog)}")
            
            try:
                interpreter_res = self.interpret(prog, unkSortMap, io_examples_tr, io_examples_val, io_examples_test)
                top_k_solutions_results.append((prog, interpreter_res))
                self.update_top_k(top_k_solutions_results)
                
                # Log validity data if enabled
                if NAS_EXPERIMENTAL_CONFIG.get("validity_check", False):
                    val_acc = interpreter_res.get('accuracy', -1.0)
                    prog_str = repr_py(prog).replace('"', '""')
                    expr = nas_metrics.get('expressivity', 0)
                    prog_metric = nas_metrics.get('progressivity', 0)
                    train = nas_metrics.get('trainability', 0)
                    norm_expr = nas_metrics.get('norm_expr', 0)
                    norm_prog = nas_metrics.get('norm_prog', 0)
                    norm_train = nas_metrics.get('norm_train', 0)
                    with open("az_nas_validity_data.csv", "a") as f:
                        f.write(f'{score},{param_count},{expr},{prog_metric},{train},{norm_expr},{norm_prog},{norm_train},{val_acc},"{prog_str}"\n')

            except NotHandledException:
                self.log_unhandled_program(prog)
            except Exception as e:
                self.log_evaluator_exception(e, prog, unkSortMap)

        # Log evaluated programs for consistency with original class
        for _, _, _, prog, _ in top_candidates:
             self.evaluated_programs_str.append(repr_py(prog))
             self.evaluated_programs_type_info.append(str(prog))

        self._log_evaluated_programs()
        
        return NeuralSynthesizerResult(top_k_solutions_results)
