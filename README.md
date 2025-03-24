# HOUDINI - Lifelong Learning
Code for running the experiments, described in our NeurIPS 2018 paper "HOUDINI: Lifelong Learning as Program Synthesis" (https://arxiv.org/abs/1804.00218)

Disclaimer: This is a new version, updated on March 2025 in order to run using the newest pytorch version.
The changes were small so they shouldn't affect the code's behaviour, but I have not re-run the full experiments to verify this.

## Requirements:
Evaluated on a macOS conda environemnt with Python 3.13.2 and the following python packages:

- torch==2.6.0
- torchaudio==2.6.0
- torchvision==0.21.0
- matplotlib==3.10.1
- numpy==2.2.4
- pillow==11.1.0
- scikit-image==0.25.2
- scipy==1.15.2
- tqdm==4.67.1


## Datasets
The datasets are automatically downloaded when needed.

## Running

### Houdini
    For the counting sequences 1, 2, 3, and the long sequence:
    usage: python HOUDINI/Eval/CS_LS.py [-h] [--synthesizer {enumerative,evolutionary}]
                                 --taskseq {cs1,cs2,cs3,ls} [--dbg]
    
    For graph the sequences:
    usage: python HOUDINI/Eval/GraphSeq.py [-h] [--synthesizer {enumerative,evolutionary}]
                                 --taskseq {gs1, gs2} [--dbg]

    optional arguments:
      -h, --help            show this help message and exit
      --synthesizer {enumerative,evolutionary}
                            Synthesizer type. (default: enumerative)
      --taskseq {cs1,cs2,cs3,ls}
                            Task Sequence
      --dbg                 If set, the sequences run for a tiny amount of data



### Baselines

Counting Sequence 1, Standalone: `python Baselines/CountingSeqLongSeq.py cs1 sa`

Counting Sequence 1, low-level-transfer: `python Baselines/CountingSeqLongSeq.py cs1 wt`

Counting Sequence 2, Standalone: `python Baselines/CountingSeqLongSeq.py cs2 sa`

Counting Sequence 2, low-level-transfer: `python Baselines/CountingSeqLongSeq.py cs2 wt`

Counting Sequence 3, Standalone: `python Baselines/CountingSeqLongSeq.py cs3 sa`

Counting Sequence 3, low-level-transfer: `python Baselines/CountingSeqLongSeq.py cs3 wt`


Long Sequence, Standalone: `python Baselines/CountingSeqLongSeq.py ls sa`

Long Sequence, low-level-transfer: `python Baselines/CountingSeqLongSeq.py ls wt`

## Project Structure

 * HOUDINI - the main code for the method
   * Eval - code for evaluating HOUDINI.
     * Task.py - Defines a Task which can be run to obtain a TaskResult object that contains information about the method's performance.
     * TaskSeq.py - Defines a Sequence of Tasks. You can run a specific task from the sequence, given an index. The run method then loads/saves a library of modules as necessary, and saves the results to a report file.
   * Interpreter
     * Interpreter.py Contains the code for training and evaluating neural networks.
     * NeuralModules.py Defines the neural modules used in our experiments.
   * Synthesizer - Contains all the program synthesis code. 
     * SymbolicSynthesizer.py - Defines the program synthesis method used in HOUDINI.
     * SymbolicSynthesizerEA.py - Defines an alternative program synthesis method based on evolutionary algorithms.
   * Tests - Contains tests for the program synthesis code.
   * FnLibrary.py - Defines a library containing higher-order functions and learned parametric functions (neural modules).
   * FnLibraryFunctions.py - Implementations of the higher-order functions.
   * NeuralSynthesizer.py - Defines the code for solving a problem using HOUDINI. The function solve() takes in training, validation and test datasets. It then synthesizes and evaluates a number of programs, while keeping track of the top k best performing programs.
  * Data - Code for generating and loading data
   * DataGenerator.py - python classes for iterating over data when training.
   * DataProvider.py - code for downloading different image datasets + utility functions for loading these images and using them to generate counting/summing tasks.
   * MazeGenerator.py - used to generate data for shortest path task on a grid of images.
 * Baselines (Acronyms: wt: low-level transfer; sa: standalone)
   * PNN - Progressive Neural Networks
   * CountingSeqLongSeq.py - Run baselines on counting sequences and the long sequence
   * GSi_[baseline].py - Run [baseline] on Graph Sequence i
   * Summing Sequence.py - Run baselineso on the summing sequence
* Results - Running HOUDINI on a sequence generates a sequence-specific folder which contains an HTML file. The HTML file contains details of HOUDINI's performance on that sequence.