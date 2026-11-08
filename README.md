# GT05 — Dynamic Spectrum Allocation in Cognitive Radio Networks

**Final year project | Genetic Algorithms and Machine Learning | 2024–25**

This project explores how wireless communication channels can be used more efficiently and fairly in **cognitive radio networks**. It compares Genetic Algorithm (GA) experiments with four machine learning classifiers using channel-availability and reward matrices.

The accompanying report is titled *Dynamic Spectrum Allocation in Cognitive Radio Networks Using Deep Learning*. The code included here implements **genetic algorithms and classical machine learning**; deep learning and reinforcement learning are discussed in the report as part of the broader project direction.

## The problem in simple terms

Wireless devices share a limited number of communication channels. Some channels may be unavailable, and available channels may provide different benefits to different users. Cognitive radio aims to adapt channel usage to these conditions.

For example, assigning the best channels to only a few users could produce a high total reward while leaving other users with poor service. This project investigates the balance between **overall performance** and **fairness across users** through matrix-based experiments.

## What the project evaluates

| Objective | Full name | What it tries to improve |
| --- | --- | --- |
| MSR | Max-Sum-Reward | The total reward across users and channels. |
| MMR | Max-Min-Reward | The lowest reward, so the least-served user or eligible entry receives a better outcome, depending on the experiment. |
| MPF | Max-Proportional-Fair | A balance between high rewards and fairness, using logarithmic or geometric-mean scoring in the notebooks. |

Two families of methods are explored:

- **Genetic Algorithms:** generate candidate reward matrices, evaluate their fitness, and improve them through selection, crossover, and mutation.
- **Machine learning classifiers:** Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). These predict high-reward versus low-reward entries and evaluate the resulting reward matrices.

Most experiments use **10 × 10, 15 × 15, and 20 × 20** subsets of the supplied 100 × 100 matrices. Some notebooks also include additional sizes, repeated trials, and averaged results.

## Repository contents

```text
Dynamic-Spectrum-Allocation/
├── README.md
└── GT05/
    ├── README.md
    └── GT05 final yr/
        ├── channel_matrix (2).csv
        ├── reward_matrix (1).csv
        ├── msr_avg.ipynb
        ├── MAXMIN_reward.ipynb
        ├── MPF.ipynb
        ├── new_Ml_MSR.ipynb
        ├── ML,MMR.ipynb
        ├── MLMPF.ipynb
        ├── GT05 final report.pdf
        ├── GT05 ppt.pptx
        └── Research paper.pdf
```

### Notebook guide

| Notebook | Method | Contents |
| --- | --- | --- |
| [msr_avg.ipynb](GT05/GT05%20final%20yr/msr_avg.ipynb) | GA | Total-reward optimization, repeated runs, and reward-versus-generation plots. |
| [MAXMIN_reward.ipynb](GT05/GT05%20final%20yr/MAXMIN_reward.ipynb) | GA | Minimum-reward experiments, including entry-based and per-user row-sum variants. |
| [MPF.ipynb](GT05/GT05%20final%20yr/MPF.ipynb) | GA | Proportional-fairness experiments using sums of logarithms of eligible rewards. |
| [new_Ml_MSR.ipynb](GT05/GT05%20final%20yr/new_Ml_MSR.ipynb) | ML | Total-reward experiments and comparisons between classifiers. |
| [ML,MMR.ipynb](GT05/GT05%20final%20yr/ML%2CMMR.ipynb) | ML + CVXPY | Classifier comparisons followed by max-min optimization with exactly three selected channels per user. |
| [MLMPF.ipynb](GT05/GT05%20final%20yr/MLMPF.ipynb) | ML | Classifier comparisons using the geometric mean of per-user reward sums. |

**Suggested starting point:** open `new_Ml_MSR.ipynb` and try the **MSR 10×10** comparison section. Then explore the corresponding GA notebook and the fairness objectives. The notebooks contain separate experiment sections; they do not form a single application pipeline.

## Dataset

Both CSV files are numeric, contain **100 rows and 100 columns**, and have **no header row**.

| File | Values | Meaning |
| --- | --- | --- |
| `channel_matrix (2).csv` | `0` or `1` | Channel status for each user–channel pair: unavailable or available. |
| `reward_matrix (1).csv` | Integers from `1` to `100` | Reward associated with each corresponding user–channel pair. |

Rows represent users and columns represent channels. Matching positions in the two matrices refer to the same user–channel pair. The notebooks commonly normalize rewards by dividing by `100.0`.

The repository supplies the matrices but does not include a separate dataset-generation script or detailed provenance information.

## How the experiments work

1. Load the channel and reward matrices.
2. Select a smaller subset, either from the top-left corner or from a randomly chosen location.
3. Run the selected method:
   - **GA:** evolve candidate matrices to increase the chosen objective score.
   - **ML:** flatten the subset, use channel values as the input feature, and create binary labels from normalized rewards, commonly using a threshold of `0.5`. Train classifiers and use their predictions to retain or zero out rewards.
4. Calculate MSR, MMR, or MPF using the selected section's implementation.
5. Display scores, matrices, and plots; some sections average results over several trials or subsets.

## Run with Google Colab

The notebooks were written for Google Colab and currently load data from Google Drive.

1. Open the desired `.ipynb` file in Google Colab.
2. In your Google Drive, create a folder named `finalyr` inside **My Drive**.
3. Upload **copies** of the two CSV files into that folder and name those copies:
   - `channel_matrix.csv`
   - `reward_matrix.csv`
4. Run this installation cell before the experiment:

   ```python
   %pip install numpy pandas matplotlib scikit-learn deap cvxpy scipy
   ```

5. Run the notebook's Google Drive mount cell and authorize access to your Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

6. Confirm that the CSV paths used by the notebook match:

   ```text
   /content/drive/MyDrive/finalyr/channel_matrix.csv
   /content/drive/MyDrive/finalyr/reward_matrix.csv
   ```

7. Run the setup cells and the experiment section you want to explore. Start with a 10 × 10 section before running larger or repeated experiments.

## Run locally with Jupyter

You need Python 3 and pip. The repository does not specify pinned package versions or a verified Python version.

From the repository root, run these commands in a terminal:

```powershell
cd GT05
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install jupyterlab numpy pandas matplotlib scikit-learn deap cvxpy scipy
.\.venv\Scripts\python.exe -m jupyter lab
```

Open a notebook inside `GT05/GT05 final yr` and make these adjustments in your working copy:

1. Skip the `google.colab` import and `drive.mount(...)` cells.
2. Replace **every** Google Drive CSV path in the sections you plan to run with the matching local filename:

   ```python
   import pandas as pd

   # These relative paths assume the kernel's working directory is GT05 final yr.
   C_full = pd.read_csv('channel_matrix (2).csv', header=None).values
   R_full = pd.read_csv('reward_matrix (1).csv', header=None).values

   assert C_full.shape == R_full.shape == (100, 100)
   ```

   Some sections use `np.loadtxt` instead of `pd.read_csv`; update those paths too. If the kernel starts in another directory, use absolute paths to the CSV files.

3. Select the virtual environment's Python kernel and run the required setup and experiment cells.

`ML,MMR.ipynb` uses binary optimization variables and needs a mixed-integer solver supported by CVXPY. The installation above includes SciPy; if a solver error occurs, check `cp.installed_solvers()` and ensure your CVXPY/SciPy installation provides a compatible mixed-integer solver.

## Expected outputs and interpretation

Depending on the section, the notebook displays:

- Original and optimized or predicted reward matrices.
- Objective scores for each generation or sampled subset.
- Average scores across trials or subsets.
- Line plots of GA progress and classifier results, and bar charts comparing classifier averages.

Higher scores indicate a better result **within the same objective, formula, scale, and experiment setup**. A score from one notebook should not automatically be compared with a score from another.

### Experimental considerations

- **Scoring formulas vary.** GA MMR includes both the minimum eligible entry and the minimum per-user row sum. ML MMR optimizes channel selections with a three-channel-per-user constraint. GA MPF sums logarithms of eligible entries, while ML MPF uses a geometric mean of per-user sums. These are different formulations.
- **The methods operate differently.** Several GA sections evolve reward values themselves, while ML sections filter the supplied rewards. Some GA mutations do not explicitly enforce the initial reward bounds. Keep these differences in mind when interpreting performance comparisons.
- **ML reward plots are not held-out accuracy measurements.** Several sections create a training/test split but predict on the full subset to calculate reward scores. Separate held-out evaluation is needed to measure generalization.
- **Results can change between runs.** Random subset selection and GA operations are not consistently seeded. For repeatable experiments, seed both NumPy and Python's `random` module before running a section.
- **Notebook state matters.** Experiment sections redefine functions and DEAP classes such as `FitnessMax` and `Individual`. Restart the kernel/runtime when switching independent experiments if you encounter redefinition warnings or unexpected state.

## Troubleshooting

| Issue | What to check |
| --- | --- |
| `FileNotFoundError` | Match the CSV names and paths exactly, including spaces and `(1)` / `(2)` when using the original local files. In Colab, mount Drive first. |
| `ModuleNotFoundError` | Run the installation command in the same Python environment as the notebook kernel. |
| `google.colab` cannot be imported locally | Skip the Colab mount cell and use local data paths. |
| CVXPY solver error in ML MMR | Check installed mixed-integer solver support and the optimization status. |
| DEAP class redefinition warnings | Restart the kernel and run only the setup and experiment section you need. |
| Long runtimes or very large output | Start with a smaller subset and reduce trial/generation counts while exploring. |

## Project documents

- [Final report](GT05/GT05%20final%20yr/GT05%20final%20report.pdf): background, methodology, results, and conclusions.
- [Presentation](GT05/GT05%20final%20yr/GT05%20ppt.pptx): project slides.
- [Research paper](GT05/GT05%20final%20yr/Research%20paper.pdf): accompanying research material.

## Project team

As listed in the final report:

- K. Jagadish
- R. Jyoshna
- B. Jyothi
- Rithika J. Poojari
- B. Karthik Goud
- Mohith Kumar

**Guide:** Dr. A. Kiran Kumar

**Institution:** Malla Reddy University, Hyderabad

**Academic year:** 2024–25
