# 🇱🇰 SinhalaMMLU
**SinhalaMMLU** is a benchmark dataset for evaluating **multitask language understanding** in **Sinhala**.  
It aims to measure the performance of multilingual and low-resource LLMs on diverse academic and cultural domains.

## 📘 Overview

| Feature | Description |
|----------|-------------|
| **Language** | Sinhala |
| **Format** | Multiple-choice questions (MCQs) |
| **Entries** | 7,044 |
| **Subjects** | 30 (Humanities, Social Science, STEM, Language, Culture, etc.) |
| **Difficulty Levels** | Easy / Medium / Hard |


### 📚 Subjects by Domain

The SinhalaMMLU dataset includes subjects categorized under six main domains, as shown below.

| **Domain** | **Subjects** |
|-------------|--------------|
| **Humanities** | History, Drama and Theatre, Dancing, Eastern Music, Arts, Buddhism, Catholicism, Christianity, Islam, Buddhist Civilization, Oriental Music, History of Sri Lanka, Dancing Indigenous |
| **Social Science** | Citizenship Education, Health and Physical Science, Geography, Political Science |
| **STEM** | Physics, Chemistry, Biology, Science |
| **Language** | Sinhala Language and Literature |
| **Business Studies** | Business and Accounting Studies, Entrepreneurship Studies, Economics |
| **Other** | Home Economics, Biosystems Technology, Communication and Media Studies, Design and Construction Technology, Agriculture and Food Technology |

**Table 1:** Subjects categorized by domain in the SinhalaMMLU dataset.

### 📊 Dataset Statistics

The following table shows the total number of questions and the average question and answer lengths (in characters) for each difficulty level and domain.

| **Group** | **# Questions** | **Question Length** | **Answer Length** |
|------------|----------------:|--------------------:|------------------:|
| **Easy** | 1893 | 59.08 | 16.77 |
| **Medium** | 2585 | 100.66 | 24.79 |
| **Hard** | 2566 | 116.40 | 27.53 |
|------------|----------------|--------------------|------------------|
| **STEM** | 629 | 157.82 | 27.42 |
| **Social Science** | 1084 | 141.80 | 22.34 |
| **Humanities** | 3419 | 93.91 | 22.24 |
| **Language** | 397 | 74.19 | 25.65 |
| **Business Studies** | 477 | 173.39 | 32.99 |
| **Other** | 1038 | 108.58 | 28.24 |

**Table 1:** Total number of questions and average question and answer length (in characters) for each difficulty level and domain.  
*The overall question count is 7,044.*


## Evaluation
The code used for evaluating each model is located in the **`src/`** directory, and the scripts to run these evaluations are provided in the **`scripts/`** directory.


## How to cite


