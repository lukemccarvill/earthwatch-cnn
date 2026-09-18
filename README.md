# Earthwatch CNN / RQ1 Water Quality Prediction Project

Sandpit project for the ILESLA programme, carried out by Luke, Becky, Grace, Pranay and Havana. Completed over four scattered weeks between October 2025 and September 2026.

This repository was created for Earthwatch UK and focuses on citizen-science water quality monitoring using WaterBlitz data. The work here is limited to RQ1.

## Project purpose

This is a research/reference repo for early CNN (imagery), MLP (metadata), and multimodal (combined imagery+metadata) experiments and exploratory modelling. The main notebook for the multimodal workflow is:

- `src/03_metadata_baseline.ipynb`

## Important data note

The WaterBlitz image folder is not stored in this repository because it is too large (~4.3 GB). To run the full analysis, access to the Earthwatch WaterBlitz image set must be obtained externally, and then the path in the code must be updated to reflect where it is stored locally on one's computer.

## Repository contents

- `src/`: notebooks and scripts for metadata baselines, CNN attempts, feature importance, multimodal experiments, and model architecture exploration
- `data/`: project datasets and metadata, with the key metadata in `merged_metadata.xlsx`
- `outputs/`: some generated model/feature outputs, though the multimodal results are simply in the jupyter notebook itself.

This repo is slightly messy and not a polished end-to-end pipeline. It mainly serves as a record of what was tried in statistics analysis and model design for future reference.

## High-level summary

- Objective: explore multimodal prediction for water quality using citizen-science observations and imagery
- Scope: RQ1 only
- Domain: Earthwatch UK / WaterBlitz
- Status: exploratory and archival reference material
