# Multimodal Fusion of Maintenance-Log Text and Sensor Time-Series Data for Improved Remaining-Useful-Life Prediction

## Abstract (draft)

Predictive maintenance (PdM) research has largely treated two rich data sources — numeric sensor time-series and free-text maintenance logs — as separate problems, despite growing evidence that each captures failure signals the other misses. Recent work has begun structuring maintenance-log text using large language models (LLMs) and generative transformers, but these efforts stop at extraction and structuring; none report a measured improvement in downstream failure-prediction accuracy from combining the extracted text features with sensor data. This paper proposes a fusion framework that combines a sensor-based Remaining Useful Life (RUL) prediction model with text-derived features from synthetically generated maintenance-log narratives, evaluated on the NASA C-MAPSS turbofan degradation benchmark. We compare a sensor-only baseline (LSTM/CNN) against a fusion model (sensor encoder + text embedding encoder) to quantify whether, and by how much, adding log-text features improves RUL prediction accuracy. The study aims to provide one of the first controlled, quantified answers to a gap explicitly flagged as open in recent literature.

**Keywords:** predictive maintenance, multimodal fusion, remaining useful life, large language models, maintenance logs, sensor time-series

---

## 1. Research Gap Statement

Across the recent literature on AI-driven maintenance-log analysis, a consistent pattern emerges: studies focus on *structuring* or *cleaning* unstructured text (extracting failure modes, causes, or corrective actions), and explicitly defer the *fusion* of that structured text with quantitative sensor data to future work, without carrying it out. Two independent 2025–2026 studies name this gap directly:

- A wind-turbine reliability study using LLMs for semantic analysis of maintenance logs states that integrating its extracted semantic insights with time-series sensor data to enhance failure-prediction accuracy is left as future work, not yet attempted [1].
- A manufacturing-sector study that structures equipment maintenance text using BART, T5, and Qwen models similarly proposes combining the resulting structured text with time-series analysis for predictive maintenance as a future research direction, rather than a completed contribution [2].

A third related study focuses only on cleaning noisy automotive maintenance logs with LLM agents to improve downstream ML training data quality, but does not evaluate any fused sensor+text prediction model, and relies on synthetic logs due to proprietary data constraints — establishing synthetic log generation as an accepted, reproducible methodology for this kind of study [3].

**This leaves an open, well-defined gap:** no controlled study has built a fused sensor + maintenance-log-text model and measured the resulting change in RUL/failure-prediction accuracy against a sensor-only baseline. This paper's contribution is to fill exactly that gap.

## 2. Related Work (draft paragraph for Literature Review section)

Recent research on applying AI to maintenance logs falls into three broad strands. The first uses LLMs for deep semantic analysis of log text — failure-mode identification, causal-chain inference, and data-quality auditing — treating the LLM as a "reliability co-pilot" for human engineers, but stopping short of feeding these insights into a quantitative prediction model [1]. The second strand applies generative transformer models (BART, T5, Qwen) to convert unstructured maintenance text into structured fields such as failed components, failure types, and corrective actions, validated on a large industrial dataset, again identifying fusion with sensor time-series as future work rather than a delivered result [2]. The third strand addresses data quality itself, using LLM agents to clean noisy, error-prone maintenance logs (typos, duplicate entries, missing fields) prior to downstream ML use, evaluated on synthetic logs due to the sensitivity of real proprietary automotive data [3]. Sensor-only RUL prediction, by contrast, is a mature area, with the NASA C-MAPSS turbofan degradation dataset serving as a standard benchmark for LSTM- and CNN-based prognostics models [4]. No study identified in this review reports a controlled comparison between a sensor-only RUL model and a fused sensor+text model on a shared benchmark — the specific gap this paper addresses.

## 3. Proposed Methodology (summary)

1. Use NASA C-MAPSS sensor time-series data as the base benchmark [4].
2. Generate synthetic maintenance-log narratives tied to known failure events, following the synthetic-data approach validated in [3].
3. Train a sensor-only baseline (LSTM/CNN) for RUL prediction.
4. Train a fusion model combining the sensor encoder with a text embedding encoder (e.g., BERT) over the synthetic logs.
5. Compare RUL prediction accuracy (RMSE, early-failure detection precision) between baseline and fusion model.

## 4. References (IEEE format)

[1] M. Malyi, J. Shek, and A. Biscaya, "Exploratory Semantic Reliability Analysis of Wind Turbine Maintenance Logs using Large Language Models," arXiv:2509.22366, Aug. 2025.

[2] Y. Cho, "Automated Structuring and Analysis of Unstructured Equipment Maintenance Text Data in Manufacturing Using Generative AI Models: A Comparative Study of Pre-Trained Language Models," *Applied Sciences*, vol. 16, no. 4, p. 1969, Feb. 2026, doi: 10.3390/app16041969.

[3] V. Dimidov, F. Hawlader, S. Jafarnejad, and R. Frank, "Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance," arXiv:2511.05311, Nov. 2025.

[4] A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage propagation modeling for aircraft engine run-to-failure simulation," in *Proc. Int. Conf. on Prognostics and Health Management (PHM08)*, Denver, CO, USA, 2008.

---

*Note: References [1]–[3] are used here to establish the research gap and should be cited in your Introduction/Related Work section. You will need to add further references as you build out the full literature review (aim for 15–25 references in a typical undergraduate paper). [4] is the standard dataset citation — cite it wherever you describe your data source.*
