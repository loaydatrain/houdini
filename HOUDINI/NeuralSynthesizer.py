import logging
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from Data.DataGenerator import NumpyDataSetIterator

# from HOUDINI.Data.DataProvider_old import split_into_train_and_validation, get_batch_count_iseven
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.InterpreterFilters import is_evaluable
from HOUDINI.FnLibraryFunctions import NotHandledException
from HOUDINI.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.ASTUtils import deconstruct
from HOUDINI.Synthesizer.MiscUtils import getElapsedTime, formatTime
from HOUDINI.Synthesizer.ReprUtils import repr_py
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer

"""
New neural synthesizer, introduced on May 4th.
"""

NeuralSynthesizerSettings = NamedTuple("NeuralSynthesizerSettings", [
    ('N', int),  # Generate at most N programs.
    ('M', int),  # Evaluate at most M programs.
    ('K', int),  # Return top-k programs
    ('progressive_tuning_schedule', Optional[List[Dict[str, Any]]]),  # Progressive tuning stages
])

ProgressiveTuningStage = NamedTuple("ProgressiveTuningStage", [
    ('train_fraction', float),
    ('max_candidates', Optional[int]),
    ('epochs', Optional[int])
])


class NSDebugInfo:
    def __init__(self, dprog):
        self.dprog = dprog


class _CandidateState:
    def __init__(self, prog, unkSortMap):
        self.prog = prog
        self.unkSortMap = unkSortMap
        self.latest_result = None


def _debug_info(prog: PPTerm, unkSortMap, lib: FnLibrary, fnSort: PPSort):
    lib_items = [PPLibItem(li.name, li.sort, None) for (_, li) in lib.items.items()]
    dprog = """
    io_examples_tr, io_examples_val = None, None
    prog = %s
    unkSortMap = %s
    lib = NewLibrary()
    lib.addItems(%s)
    fn_sort = %s
    interpreter = Interpreter(lib, epochs=1)
    res = interpreter.evaluate(program=prog,
                                        output_type_s=fn_sort.rtpe,
                                        unkSortMap=unkSortMap,
                                        io_examples_tr=io_examples_tr,
                                        io_examples_val=io_examples_val)
    """ % (str(prog), str(unkSortMap), str(lib_items), str(fnSort))

    return NSDebugInfo(dprog),


class NeuralSynthesizerResult:
    def __init__(self,
                 top_k_solutions_results: List[Tuple[PPTerm, Dict]]):
        # A list of top scoring programs
        self.top_k_solutions_results = top_k_solutions_results

    def get_top_solution_score(self):
        """
        Top scoring program and corresponding score
        """
        top_solution_score = self.top_k_solution_scores[0] if len(self.top_k_solution_scores) else None
        return top_solution_score


def get_lib_names(term: PPTerm) -> List[str]:
    if isinstance(term, PPVar):
        name = term.name
        if name[:4] == "lib.":
            return [name[4:]]
        else:
            return []
    else:
        nts = []
        for c in deconstruct(term):
            cnts = get_lib_names(c)
            nts.extend(cnts)
        return nts


class NeuralSynthesizer:
    def __init__(self, interpreter: Interpreter, synthesizer: SymbolicSynthesizer,
                 lib: FnLibrary, sort: PPFuncSort,
                 dbg_learn_parameters,
                 settings
                 ):
        self.interpreter = interpreter
        self.synthesizer = synthesizer
        self.lib = lib
        self.sort = sort
        self.settings = settings
        self.prog_unkinfo_tuples = []
        self.progressive_schedule = self._normalize_progressive_schedule(settings.progressive_tuning_schedule)
        self.rejection_counts = defaultdict(int)
        self.total_rejections = 0
        self._needs_rejection_newline = False

        self.dbg_learn_parameters = dbg_learn_parameters

        self.evaluated_programs_str = []
        self.evaluated_programs_type_info = []

        self.init_progs()

    def _normalize_progressive_schedule(self, schedule_data):
        if not schedule_data:
            return []

        normalized = []
        for raw_stage in schedule_data:
            if isinstance(raw_stage, ProgressiveTuningStage):
                normalized.append(raw_stage)
                continue

            if isinstance(raw_stage, dict):
                train_fraction = raw_stage.get('train_fraction', 1.0)
                max_candidates = raw_stage.get('max_candidates')
                epochs = raw_stage.get('epochs')
            else:
                # assume tuple-like ordering
                train_fraction, max_candidates, epochs = raw_stage

            train_fraction = float(train_fraction)
            train_fraction = max(0.01, min(1.0, train_fraction))
            if max_candidates is not None:
                max_candidates = int(max_candidates)
                if max_candidates <= 0:
                    max_candidates = None
            if epochs is not None:
                epochs = max(1, int(epochs))

            normalized.append(ProgressiveTuningStage(train_fraction, max_candidates, epochs))

        return normalized

    def _slice_io_examples(self, io_examples, fraction):
        if io_examples is None or fraction is None or fraction >= 0.9999:
            return io_examples

        fraction = max(0.0, min(1.0, fraction))

        if issubclass(type(io_examples), NumpyDataSetIterator):
            total = io_examples.inputs.shape[0]
            count = max(1, int(total * fraction))
            return io_examples.inputs[:count], io_examples.targets[:count]
        elif type(io_examples) == tuple:
            total = io_examples[0].shape[0]
            count = max(1, int(total * fraction))
            return io_examples[0][:count], io_examples[1][:count]
        elif type(io_examples) == list:
            return [self._slice_io_examples(item, fraction) for item in io_examples]
        else:
            return io_examples

    def init_progs(self):
        n = 0
        m = 0
        pStart = time.time()
        print('BEGIN_PROGRAM_GENERATION, Time: %s' % getElapsedTime())
        for prog, unkSortMap in self.synthesizer.genProgs():
            # print(m,n)
            n += 1
            if n % 100 == 0:
                print('.', end='', flush=True)

            if n > self.settings.N or m >= self.settings.M:
                break

            c_program_str_representation = repr_py(prog)

            try:
                is_ok, ecode = is_evaluable(prog, self)
            except Exception as e:
                self.log_isevaluable_exception(e, prog, unkSortMap)
                continue

            if is_ok:
                self.evaluated_programs_str.append(c_program_str_representation)
                self.evaluated_programs_type_info.append(str(prog))
                self.prog_unkinfo_tuples.append((prog, unkSortMap))
                m += 1
            else:
                if ecode != 2:
                    self.log_rejected_program(prog, ecode)

        if self._needs_rejection_newline:
            print()
        print('END_PROGRAM_GENERATION, Time: %s' % getElapsedTime())
        pEnd = time.time()
        print("TIME_TAKEN_SYNTH, %s" % formatTime(pEnd - pStart))

    def interpret(self, prog, unkSortMap, io_examples_tr, io_examples_val, io_examples_test) -> Dict:
        output_type = self.sort.rtpe
        print('BEGIN_EVALUATE, Time: %s' % getElapsedTime())
        eStart = time.time()
        res = self.interpreter.evaluate(program=prog,
                                        output_type_s=output_type,
                                        unkSortMap=unkSortMap,
                                        io_examples_tr=io_examples_tr,
                                        io_examples_val=io_examples_val,
                                        io_examples_test=io_examples_test,
                                        dbg_learn_parameters=self.dbg_learn_parameters)
        print('END_EVALUATE, Time: %s' % getElapsedTime())
        eEnd = time.time()
        print("TIME_TAKEN_EVALUATE, %s" % formatTime(eEnd - eStart))

        return res

    def update_top_k(self, top_k_solutions_results: List[Tuple[PPTerm, Dict]]):
        top_k_solutions_results.sort(key=lambda x: x[1]['accuracy'], reverse=True)
        if len(top_k_solutions_results) > self.settings.K:
            del top_k_solutions_results[-1]

        for i in range(1, top_k_solutions_results.__len__()):
            top_k_solutions_results[i][1]["new_fns_dict"] = None

    def solve(self, io_examples_tr, io_examples_val, io_examples_test) -> List[Tuple[PPTerm, float]]:
        if self.progressive_schedule:
            result = self._solve_progressive(io_examples_tr, io_examples_val, io_examples_test)
        else:
            result = self._solve_single_pass(io_examples_tr, io_examples_val, io_examples_test)

        self._log_evaluated_programs()
        return result

    def _solve_single_pass(self, io_examples_tr, io_examples_val, io_examples_test):
        top_k_solutions_results = []
        for prog, unkSortMap in self.prog_unkinfo_tuples:
            try:
                interpreter_res = self.interpret(prog, unkSortMap, io_examples_tr, io_examples_val, io_examples_test)
            except NotHandledException:
                self.log_unhandled_program(prog)
                continue
            except Exception as e:
                e.args += _debug_info(prog, unkSortMap, self.lib, self.sort)
                traceback.print_exc()
                self.log_evaluator_exception(e, prog, unkSortMap)
                continue

            top_k_solutions_results.append((prog, interpreter_res))
            self.update_top_k(top_k_solutions_results)

        return NeuralSynthesizerResult(top_k_solutions_results)


# (PPFuncApp≈(fn=PPVar(name='lib.compose'), args=[PPTermUnk(name='nn_fun_cs1_d0d1_np_tdr0_2', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=64), PPDimConst(value=4), PPDimConst(value=4)])], rtpe=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)]))), PPTermUnk(name='nn_fun_cs1_d0d1_np_tdr0_3', sort=PPFuncSort(args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)])], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=64), PPDimConst(value=4), PPDimConst(value=4)])))]), 
# {'nn_fun_cs1_d0d1_np_tdr0_2': PPFuncSort(args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=64), PPDimConst(value=4), PPDimConst(value=4)])], rtpe=PPTensorSort(param_sort=PPBool(), shape=[PPDimConst(value=1), PPDimConst(value=1)])), 'nn_fun_cs1_d0d1_np_tdr0_3': PPFuncSort(args=[PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=1), PPDimConst(value=28), PPDimConst(value=28)])], rtpe=PPTensorSort(param_sort=PPReal(), shape=[PPDimConst(value=1), PPDimConst(value=64), PPDimConst(value=4), PPDimConst(value=4)]))})

    def _solve_progressive(self, io_examples_tr, io_examples_val, io_examples_test):
        if not self.prog_unkinfo_tuples:
            return NeuralSynthesizerResult([])

        # for t in self.prog_unkinfo_tuples:
        #     print(t)
        # sys.exit()

        candidates = [_CandidateState(prog, unkSortMap) for prog, unkSortMap in self.prog_unkinfo_tuples]
        active_candidates = candidates
        original_epochs = self.interpreter.epochs

        try:
            for stage_idx, stage in enumerate(self.progressive_schedule):
                print("\n"*3, "="*50)
                print("BEGIN_PROGRESSIVE_STAGE %d: fraction=%.2f, epochs=%s, survivors=%s" % (
                    stage_idx,
                    stage.train_fraction,
                    stage.epochs if stage.epochs is not None else self.interpreter.original_num_epochs,
                    stage.max_candidates if stage.max_candidates is not None else 'ALL'))
                stage_train_examples = self._slice_io_examples(io_examples_tr, stage.train_fraction)
                stage_epochs = stage.epochs if stage.epochs is not None else self.interpreter.original_num_epochs
                self.interpreter.epochs = stage_epochs

                evaluated_candidates = []
                for candidate in active_candidates:
                    # print("\ncandidates", candidate.prog, active_candidates, "\n")
                    try:
                        interpreter_res = self.interpret(candidate.prog, candidate.unkSortMap,
                                                         stage_train_examples,
                                                         io_examples_val,
                                                         io_examples_test)
                    except NotHandledException:
                        self.log_unhandled_program(candidate.prog)
                        continue
                    except Exception as e:
                        e.args += _debug_info(candidate.prog, candidate.unkSortMap, self.lib, self.sort)
                        traceback.print_exc()
                        self.log_evaluator_exception(e, candidate.prog, candidate.unkSortMap)
                        continue

                    candidate.latest_result = interpreter_res
                    evaluated_candidates.append(candidate)

                evaluated_candidates.sort(key=lambda c: c.latest_result['accuracy'], reverse=True)
                print("num of evaluated candidates", len(evaluated_candidates))
                if stage.max_candidates is not None:
                    evaluated_candidates = evaluated_candidates[:stage.max_candidates]

                print("END_PROGRESSIVE_STAGE %d: surviving candidates=%d" % (stage_idx, len(evaluated_candidates)))
                active_candidates = evaluated_candidates
                if not active_candidates:
                    break
        finally:
            self.interpreter.epochs = original_epochs

        top_k_solutions_results = [(cand.prog, cand.latest_result)
                                   for cand in active_candidates if cand.latest_result is not None]
        self.update_top_k(top_k_solutions_results)
        return NeuralSynthesizerResult(top_k_solutions_results)

    def _log_evaluated_programs(self):
        print("Exiting NeuralSynthesizer.solve(). The following programs were evaluated:")
        for idx, program_str in enumerate(self.evaluated_programs_str):
            print(program_str)
            print(self.evaluated_programs_type_info[idx])
            print("..........................")

    def log_evaluated_program(self, prog):
        print("Program evaluated: %s" % repr_py(prog))

    # def log_rejected_program(self, prog, ecode):
    #     print("Program rejected (ecode %d): %s" % (ecode, repr_py(prog)))
    #     # print("Program rejected (pyrep): %s" % str(prog))
    def log_rejected_program(self, prog, ecode):
        self.total_rejections += 1
        self.rejection_counts[ecode] += 1
        stats_str = ', '.join(
            f"{code}:{count}" for code, count in sorted(self.rejection_counts.items(), key=lambda kv: kv[0]))
        bar = f"Rejected programs: {self.total_rejections} | ecode distribution [{stats_str}]"
        print('\r' + bar, end='', flush=True)
        self._needs_rejection_newline = True
        # keep repr_py call to preserve side-effect expectations if needed


    def log_unhandled_program(self, prog):
        print("Program not handled: %s" % repr_py(prog))

    def log_evaluator_exception(self, e, prog, unkSortMap):
        loggerE = logging.getLogger('pp.exceptions')
        loggerE.error('Exception in the Interpreter.\n %s' % repr(e))
        loggerE.error('DebugInfo.\n %s' % _debug_info(prog, unkSortMap, self.lib, self.sort)[0].dprog)

    def log_isevaluable_exception(self, e, prog, unkSortMap):
        loggerE = logging.getLogger('pp.exceptions')
        e.args += _debug_info(prog, unkSortMap, self.lib, self.sort)
        loggerE.error('#### Exception in the Interpreter.\n %s' % repr(e))
