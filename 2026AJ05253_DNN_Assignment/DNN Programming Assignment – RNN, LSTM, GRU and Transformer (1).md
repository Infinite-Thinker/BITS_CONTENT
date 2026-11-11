# **DNN Programming Assignment-2 – RNN, LSTM, GRU and Transformer**

| Deadline | Friday 14 November 2026, 11:59PM |
| :---- | :---- |
| Late window, 20% of the marks deducted | Sunday 15 November 2026, 11:59PM |

## **Overview**

One assignment, two programming problems, sized at 2–3 days of student effort (about 25–30 hours): Problem 1 builds and compares RNN, LSTM and GRU models on sequence tasks, and Problem 2 builds a Transformer from scratch.

**How the request is read.**  Problem 1 is purely recurrent (vanilla RNN, LSTM and GRU on sequences, no CNN). The Transformer is a separate problem. If you meant a different split, the structure below is easy to change.

**Learning outcomes.** A student who finishes can:

* implement RNN, LSTM and GRU cells from their equations and prove each matches the framework's version;  
* compare RNN, LSTM and GRU on real sensor sequences and a long-memory task, in accuracy, parameters, training time and gradient behaviour;  
* build scaled dot-product attention, multi-head attention, positional encoding and an encoder block from tensor operations;  
* train a small causal Transformer, inspect its attention and show what breaks when a design choice is removed;  
* report results with evidence (multiple seeds, plots, a short analysis) rather than a single number.

**Ground rules.**

* Python 3 and PyTorch. Layers such as Linear, LayerNorm and Embedding are allowed.  
* nn.LSTM, nn.GRU and nn.RNN may be used in the Problem 1 model comparison and experiments, but the LSTM cell in Problem 1 task 2 is written from scratch. nn.MultiheadAttention and nn.Transformer\* are not allowed anywhere in Problem 2\. Library versions may be used only to check a student's own code.  
* No pre-trained weights.  
* A free Colab T4 GPU is enough: each Problem 1 run takes a few minutes and the Problem 2 language model takes about 10–15 minutes.

## **Key dates and submission**

Each student submits one zip file through the course portal. A 24-hour late window follows, with a penalty. The dates in this section are proposed; confirm or change them before release.

Note: In case of any ambiguity or missing information, you may make reasonable assumptions and proceed accordingly. Clearly mention any assumptions you make.

| Milestone | Date and time (IST) |
| :---- | :---- |
| Assignment released | Sunday 9 November 2026 |
| Last day to request an alternative dataset (Problem 1 only) | Wed 11 November 2026, 11:59PM |
| Deadline | Friday 14 November 2026, 11:59PM |
| Late window, 20% of the marks deducted | Sunday 15 November 2026, 11:59PM |
| After the late window | Not accepted |

**Where.** Upload at **\[portal link – instructor to add\]**. The portal's clock decides whether a submission is on time.

**What.**

* One zip named \<StudentID\>\_DNN\_Assignment.zip, for example 2024AB12345\_DNN\_Assignment.zip. Any other name or format (rar, 7z, a folder) is treated as not submitted.  
* Maximum size 25 MB. Do not include the datasets, model checkpoints or the virtual environment.  
* A zip that does not open, or has no report.pdf inside, counts as not submitted. Download the file after uploading and check that it opens.  
* Keep the portal's confirmation screen or email as proof of submission.

**Folder structure inside the zip.**

\<StudentID\>\_DNN\_Assignment/  
  README.md  
  requirements.txt  
  report.pdf  
  results.csv  
  problem1/  
    problem1.ipynb  
    figures/  
  problem2/  
    problem2.ipynb  
    generated\_samples.txt  
    figures/

## **Datasets and environment**

Two datasets are free and download with a few lines of code and the third is generated in code; nothing else needs to be downloaded, and no pre-trained model is used.

**Dataset 1: UCI HAR (Problem 1 and Problem 2, Task A).** Official page: [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones). Download with the page's Download button, or in code:

import urllib.request, zipfile  
url \= 'https\://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'  
urllib.request.urlretrieve(url, 'har.zip')  
zipfile.ZipFile('har.zip').extractall('har')  
\# The archive may hold a second zip, 'UCI HAR Dataset.zip': extract that one too.  
After unzipping, the folder UCI HAR Dataset holds everything used:

* train/Inertial Signals/ and test/Inertial Signals/: nine text files per split, each with one row per window of 128 steps.  
* Channel order, fixed for everyone: body\_acc\_x, body\_acc\_y, body\_acc\_z, body\_gyro\_x, body\_gyro\_y, body\_gyro\_z, total\_acc\_x, total\_acc\_y, total\_acc\_z.  
* train/y\_train.txt and test/y\_test.txt: labels 1–6 (walking, walking upstairs, walking downstairs, sitting, standing, laying).  
* train/subject\_train.txt and test/subject\_test.txt: the subject ID of every window.  
* Stacked shapes: train (7352, 128, 9), test (2947, 128, 9). The 21 training subjects and 9 test subjects do not overlap.  
* Validation split: hold out 4 of the 21 training subjects, chosen with np.random.default\_rng(seed).choice(train\_subjects, 4, replace=False), using each of the student's three seeds in turn.

**Dataset 2: Tiny Shakespeare (Problem 2, Task B).** Source: [karpathy/tiny\_shakespeare](https://huggingface.co/datasets/karpathy/tiny_shakespeare). Plain-text file, about 1.1 MB:

import urllib.request  
url \= 'https\://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'  
text \= urllib.request.urlopen(url).read().decode('utf-8')  
Build the character vocabulary from the whole text (65 characters). Use the first 90% of the characters for training and the last 10% for validation, with no shuffling.

**Dataset 3 – adding problem (Problem 1, generated in code).** No download is needed. Each input has two channels: a random value in \[0, 1\) and a marker. Exactly two steps are marked, one in each half of the sequence, and the target is the sum of the two marked values. Use this function so that every student has identical data:

import numpy as np  
def make\_adding(n, T, seed):  
    rng \= np.random.default\_rng(seed)  
    x \= np.zeros((n, T, 2), dtype=np.float32)  
    x\[:, :, 0\] \= rng.random((n, T))  
    i1 \= rng.integers(0, T // 2, n)  
    i2 \= rng.integers(T // 2, T, n)  
    x\[np.arange(n), i1, 1\] \= 1  
    x\[np.arange(n), i2, 1\] \= 1  
    y \= x\[np.arange(n), i1, 0\] \+ x\[np.arange(n), i2, 0\]  
    return x, y.astype(np.float32)  
For each T use 10,000 training sequences (seed 0), 1,000 validation (seed 1\) and 1,000 test (seed 2). A model that always predicts 1.0 has MSE of about 0.167, which is the baseline to beat.

**If a download is blocked:** use the instructor's copy at **\[mirror link – instructor to add\]**.

**Environment.**

* Python 3.10 or later, PyTorch 2.x, NumPy, pandas, Matplotlib and scikit-learn. List the exact versions in requirements.txt.  
* Google Colab: Runtime → Change runtime type → T4 GPU, then check torch.cuda.is\_available() returns True. Save results.csv after every run so a disconnected session loses nothing.  
* Fix all seeds (random, numpy, torch). Small run-to-run differences on a GPU are expected and are not penalised.

## **Problem 1 – RNN, LSTM and GRU on sequences (50 points)**

You will build the three recurrent cells (vanilla RNN, LSTM, GRU) from their equations, then compare them on two sequence tasks: classifying real sensor sequences (UCI HAR) and a synthetic long-memory task (the adding problem) where the sequence length is a dial you control. The aim is to see, with evidence, why gated cells exist.

**Data.** (i) UCI HAR exactly as described under Datasets (128 steps × 9 channels, 6 classes, subject-wise split). (ii) The adding problem, generated in code (Dataset 3): each input has two channels, a random value and a marker; the target is the sum of the two marked values.

**Tasks**

1. **Data pipelines (5).** Build the HAR pipeline (subject-wise validation split, per-channel mean and std from the training part only, class-balance plot). Build the adding-problem generator with the provided function and write a test that every sequence has exactly two markers, one in each half, and that the target equals the sum of the marked values.  
2. **Cells from scratch (10).** Implement MyRNNCell (tanh), MyLSTMCell and MyGRUCell using only matrix operations (no nn.RNN, nn.LSTM, nn.GRU or their cell classes inside these). Copy the weights from the library layer and show that your outputs match nn.RNN, nn.LSTM and nn.GRU within 1e-5 on a random batch. For d \= 9 and h \= 64 derive the parameter counts by hand (RNN 4,736; LSTM 18,944; GRU 14,208), reconcile them with your code, and explain why PyTorch reports 4,800, 19,200 and 14,400 (it keeps two bias vectors).  
3. **HAR classification (10).** Train RNN, LSTM and GRU classifiers that read the 128 steps and classify from the last hidden state. Use the same hidden size, optimiser (Adam, lr 1e-3), batch size and epoch budget for all three, and 3 seeds each. Report test accuracy and macro-F1 (mean ± std), parameters and seconds per epoch, plus the confusion matrix of the best model.  
4. **Adding problem (10).** For T \= 50, 100, 200 and 400 train all three models with 3 seeds each. Report test MSE (mean ± std) against the constant-prediction baseline of about 0.167, and plot learning curves at T \= 200\. Call a run solved if test MSE is below 0.05, and tabulate how many runs per model and length are solved.  
5. **Gradients, ablations and analysis (15).**  
   * (a) At T \= 100 plot ‖∂L/∂h\_t‖ against t on a log scale for the three models, before and after training.  
   * (b) Hidden-size sweep {16, 32, 64, 128} for LSTM and GRU on HAR.  
   * (c) Individual ablation, chosen by the last digit of your student ID (see Individualisation): even \= gradient clipping (norm 1.0 versus none) for the vanilla RNN at T \= 200; odd \= LSTM forget-gate bias initialised to 0 versus 1 at T \= 200\.  
   * (d) One-page analysis: what the gradient plot says about memory, where each model fails as T grows, the accuracy–parameters–time trade-off, and which model you would put on a battery-powered wearable, with reasons taken from your tables.

**Sanity range for marking (approximate, not a target).** On HAR, LSTM and GRU typically reach about 88–93% test accuracy, with the plain RNN lower and less stable. On the adding problem, short sequences are solved by all three; as T grows the plain RNN often stalls near the 0.167 baseline, while LSTM and GRU may need gradient clipping or more epochs at T \= 200–400. Results outside these ranges are fine if the method is sound and the analysis explains them.

## **Problem 2 – Transformer from scratch (50 points)**

Students build attention and an encoder block from tensor operations, use them on two tasks, and measure what each design choice contributes.

**Tasks**

1. **Attention from scratch (10 points).** Write scaled dot-product attention with an optional mask. Unit tests must show that each row of weights sums to 1, that the output matches torch.nn.functional.scaled\_dot\_product\_attention within 1e-5, and that a causal mask gives zero weight to future positions. Include a hand-checkable 3-token example supplied by the instructor, computed by hand and by code.  
2. **Multi-head attention and encoder block (10 points).** Implement multi-head attention with W\_Q, W\_K, W\_V and W\_O and head reshaping, a pre-LayerNorm encoder block with a feed-forward layer of width 4d, residual connections and dropout, and sinusoidal positional encoding. Derive the weight count 4d² \+ 2·d·d\_ff per block and match it to sum(p.numel()). For d \= 128 and d\_ff \= 512 the weights alone come to 196,608 per block; students reconcile the biases and LayerNorm parameters.  
3. **Task A: encoder classifier on the HAR data (10 points).** Reuse the Problem 1 data. Embed each time step (or each patch of 8 steps), add a CLS token, and classify with the encoder. Compare accuracy, parameters and training time with the best Problem 1 model. Ablate positional information: none, sinusoidal and learned.  
4. **Task B: character-level causal language model (10 points).** Train on Tiny Shakespeare (about 1.1 MB, 65 distinct characters) with context 128, d \= 128, 4 heads and 4 layers. Report validation loss and bits per character. Generate 500 characters at temperature 0.2, 0.8 and 1.5 with top-k sampling, and describe the differences.  
5. **Analysis and ablations (10 points).** (a) Heatmaps of two heads on one sentence, with an interpretation. (b) Heads \= 1, 4 and 8 at fixed d. (c) Remove the causal mask and show what happens to training loss and generation. (d) Measure time and memory at context 64, 128, 256 and 512 and relate the growth to O(n²·d).

**Sanity range for marking.** A uniform guess over 65 characters gives a loss of ln 65 ≈ 4.17 nats. A small trained model typically reaches about 1.5–1.8 nats on validation. The range is approximate; the shape of the learning curve and the ablation evidence matter more.

**Optional stretch (up to 5 bonus points).** Add a key-value cache to generation and report the speed-up, or train the Problem 1 LSTM on the same character task and compare it with the Transformer.

## **Report and code requirements**

The report is a PDF of at most 6 pages (A4, 11 pt, margins at least 2 cm). The AI-use statement and references do not count toward the 6 pages.

**Report outline.**

1. **Title block:** name, student ID  
2. **Problem 1:** setup (datasets, split, preprocessing); verification of the RNN, LSTM and GRU cells (test result and parameter counts); HAR results table; adding-problem table and curves; experiments 5(a)–(c); the one-page analysis.  
3. **Problem 2:** attention and encoder-block tests; parameter count derivation; Task A table; Task B learning curve and bits per character; heatmaps and ablations; the analysis.  
4. **AI-use statement and references.**

Every figure and table is numbered, has a caption, and is referred to in the text. A claim without a figure or table behind it earns no marks.

**Figures that must appear.** Problem 1: class balance, confusion matrix of the best HAR model, adding-problem learning curves at T \= 200 and MSE against T, gradient norm against time step, hidden-size sweep, the task 5(c) ablation plot. Problem 2: two attention heatmaps, learning curves, heads ablation, no-mask result, time and memory against context length.

**Code.**

* Each notebook runs top to bottom with *Restart and Run All* and is submitted with its outputs saved.  
* Unit tests are assert cells, placed directly after the function they test.  
* Relative paths only; no paths such as /content/drive/MyDrive/... that exist only on the student's account.  
* Comments explain the tensor shapes in the attention and LSTM code.

**results.csv.** One row per run, with exactly these columns: problem, model, seed, seq\_len, test\_acc, macro\_f1, test\_mse, params, sec\_per\_epoch, val\_loss\_nats, bits\_per\_char, notes. Columns that do not apply to a run are left empty. For the language model, bits\_per\_char is the validation loss in nats divided by ln 2\.

**generated\_samples.txt.** Three labelled samples of 500 characters, one each at temperature 0.2, 0.8 and 1.5.

**README.md.** At most one page: environment, how to run each notebook, the total run time on a T4, the student's variant, and anything that does not work.

## **Marking rubric (100 points)- Scaled to standard marks after submission/evaluations** 

The assignment is out of 100 and can be scaled to whatever weightage it carries in the course. Marks go to method and evidence, not to the highest accuracy.

| Task | Points | Full marks needs |
| :---- | :---- | :---- |
| P1.1 Data pipelines | 5 | HAR subject-wise split, training-only statistics, class-balance plot; adding-problem generator with its test |
| P1.2 RNN, LSTM and GRU cells from scratch | 10 | All three cells match their library layers (e.g. nn.LSTM) within 1e-5; parameter counts explained, including the bias difference |
| P1.3 HAR classification | 10 | Same budget for all three, 3 seeds, mean ± std, parameters, time and confusion matrix reported |
| P1.4 Adding problem | 10 | T \= 50, 100, 200, 400 × 3 models × 3 seeds; MSE against the 0.167 baseline; solved-run table; curves at T \= 200 |
| P1.5 Gradients, ablations and analysis | 15 | Gradient-norm plot, hidden-size sweep, individual ablation; conclusions backed by the tables; wearable recommendation justified |
| P2.1 Attention from scratch | 10 | Tests pass; hand example and code agree |
| P2.2 Multi-head and encoder block | 10 | Correct shapes; parameter count derived and reconciled with code |
| P2.3 Encoder classifier on HAR | 10 | Fair comparison with Problem 1; positional-encoding ablation |
| P2.4 Character language model | 10 | Learning curve, bits per character, three temperatures discussed |
| P2.5 Analysis and ablations | 10 | Heatmaps interpreted; heads, no-mask and context-length studies |
| Bonus | up to 5 | Key-value cache or LSTM-versus-Transformer character model |

**Deductions.** Up to 10 points off for code that does not run from the README, results that cannot be reproduced from the fixed seeds, or a report over 6 pages. Copied code, or use of a forbidden layer in a from-scratch part, is handled under the academic-integrity policy.

## **Rules: individual work, AI tools and integrity**

The assignment is individual work. Discussing ideas with classmates is fine; sharing or copying code, results or report text is not.

* **Frameworks.** PyTorch only. TensorFlow, JAX and Keras are not accepted.  
* **Data.** Only the datasets named above (UCI HAR, Tiny Shakespeare and the generated adding problem). No external data and no pre-trained weights.  
* **From-scratch parts.** The RNN, LSTM and GRU cells (Problem 1 task 2\) and the attention, multi-head attention and encoder block (Problem 2\) must be the student's own code. Framework versions may be used only to check them.  
* **AI tools.** Allowed for explaining concepts, debugging error messages and tidying plots. Not allowed for writing the from-scratch parts. Every use must be declared in the AI-use statement at the end of the report: the tool and what it was used for. An undeclared use found at the viva or in marking is an integrity case.  
* **Checks.** Submissions are compared against each other and against public repositories. Because every student has a different variant (hidden size, width, seeds), identical numbers or plots from two students are flagged automatically.  
* **Consequence.** Copied work is handled under the institute's academic-integrity policy.

## **Frequently asked questions**

**In case of any ambiguity or missing information, you may make reasonable assumptions and proceed accordingly. Clearly mention any assumptions you make.**

| Question | Answer |
| :---- | :---- |
| Can we work in groups? | No. The assignment is individual. |
| Can I use nn.LSTM and nn.GRU? | Yes in Problem 1 tasks 3, 4 and 5 (the comparisons and experiments), where speed matters. Not in Problem 1 task 2, where you write your own cell and use nn.LSTM (and nn.RNN, nn.GRU) only to check it. |
| Can I use nn.Transformer or nn.MultiheadAttention? | No. Problem 2 uses your own attention, multi-head attention and encoder block throughout; the library versions may only check your code. |
| Can I use TensorFlow or Keras? | No. PyTorch only. |
| Do I need a GPU? | A free Colab T4 is enough. Problem 1 also runs on a laptop CPU. For the language model without a GPU, use d \= 64 and 2 layers and say so in the README. |
| Which hidden size, width and seeds are mine? | Look up the last digit of your student ID in the variants table, under Individualisation. |
| How many epochs and what learning rate? | Your choice, but the same budget for every model within a comparison. Use Adam with learning rate 1e-3 as the starting point and state the epoch count in the report. |
| Can I change the model details (layers, dropout)? | Yes: number of layers, dropout and the classifier head are yours to choose, as long as the hidden size is your variant's value and RNN, LSTM and GRU get the same training budget. |
| My accuracy is below the sanity range. | That is acceptable if the method is sound, the tests pass and your analysis explains it. Marks follow method and evidence. |
| My Colab session disconnected. | Write results.csv after every run and keep the notebook outputs. If you are short of time, reduce epochs and say so. |
| What does mean ± std mean here? | The mean and the sample standard deviation of the metric over your three seeds. |
| Can I use a different dataset? | Problem 1 only, and only with approval; send the request by Wed 11 November 2026, 11:59 IST. Problem 2 uses the datasets given. |
| Can I get an extension? | The 24-hour late window is the only extension. A documented emergency must be raised with the instructor before the deadline. |
| Will there be a viva? | No |
| Where do I ask a doubt? | Drop mails to list of LFs and keep me in CC LFs email:  fasnaaa@wilp.bits-pilani.ac.in  karsalan@wilp.bits-pilani.ac.in  rohitsriram@wilp.bits-pilani.ac.in  gourav.kalra@wilp.bits-pilani.ac.in   rpandey@wilp.bits-pilani.ac.in  kishoregr@wilp.bits-pilani.ac.in   kavitabhardwaj@wilp.bits-pilani.ac.in  aniketchauhan@wilp.bits-pilani.ac.in  murtuza.dahodwala@wilp.bits-pilani.ac.in Dr. Eht E Sham(IC) ehte.sham@pilani.bits-pilani.ac.in  |

