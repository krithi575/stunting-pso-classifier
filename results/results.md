# District child-stunting classification: results

Generated on 2026-11-01

Random Forest baseline vs Random Forest tuned with Particle Swarm Optimization (PSO), on NFHS-5 district data.

## Dataset
- Districts: 341 (272 for training, 69 for testing)
- Indicators per district: 86
- Label: high risk when child stunting is above 35.5% (148 high, 193 low)
- Removed from the inputs to prevent leakage: 5 child-nutrition outcome columns

## PSO settings
- Particles: 10
- Iterations: 12
- Independent runs (seeds): 3
- Fitness: mean stratified 5-fold cross-validated F1 on the training set

## Baseline vs tuned (held-out test set)
| Metric | Baseline | PSO-tuned | Change |
|---|---|---|---|
| Accuracy | 0.768 | 0.826 | +0.058 |
| Precision | 0.818 | 0.875 | +0.057 |
| Recall | 0.600 | 0.700 | +0.100 |
| F1 | 0.692 | 0.778 | +0.085 |
| ROC-AUC | 0.855 | 0.875 | +0.020 |
| 5-fold CV F1 (training) | 0.740 | 0.762 | +0.022 |

## Best parameters found
| Parameter | Value |
|---|---|
| n_estimators | 138 |
| max_depth | 23 |
| min_samples_split | 13 |
| min_samples_leaf | 2 |
| max_features | 0.148 |

## Each PSO run
| Run | CV F1 (training) | Test F1 | Parameters |
|---|---|---|---|
| 1 | 0.755 | 0.741 | {'n_estimators': 91, 'max_depth': 9, 'min_samples_split': 11, 'min_samples_leaf': 5, 'max_features': 0.353} |
| 2 | 0.754 | 0.692 | {'n_estimators': 112, 'max_depth': 3, 'min_samples_split': 13, 'min_samples_leaf': 4, 'max_features': 0.157} |
| 3 | 0.762 | 0.778 | {'n_estimators': 138, 'max_depth': 23, 'min_samples_split': 13, 'min_samples_leaf': 2, 'max_features': 0.148} |

## Confusion matrices (test set)
**Baseline**

|  | Predicted low | Predicted high |
|---|---|---|
| **Actual low** | 35 | 4 |
| **Actual high** | 12 | 18 |

**PSO-tuned**

|  | Predicted low | Predicted high |
|---|---|---|
| **Actual low** | 36 | 3 |
| **Actual high** | 9 | 21 |

## Most important indicators (tuned model)
| # | Indicator | Importance |
|---|---|---|
| 1 | Women whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2) (%) | 0.0900 |
| 2 | Births in a public health facility that were delivered by caesarean section (%) | 0.0715 |
| 3 | Women who are overweight or obese (BMI ≥25.0 kg/m2) (%) | 0.0386 |
| 4 | Births in the 5 years preceding the survey that are third or higher order (%) | 0.0379 |
| 5 | Births delivered by caesarean section (%) | 0.0325 |
| 6 | Children age 6-59 months who are anaemic (<11.0 g/dl) (%) | 0.0309 |
| 7 | Women age 20-24 years married before age 18 years (%) | 0.0297 |
| 8 | Population living in households that use an improved sanitation facility2 (%) | 0.0282 |
| 9 | All women age 15-19 years who are anaemic22 (%) | 0.0264 |
| 10 | Pregnant women age 15-49 years who are anaemic (<11.0 g/dl) (%) | 0.0252 |
| 11 | Women age 15-24 years who use hygienic methods of protection during their menstrual period5 (%) | 0.0249 |
| 12 | Blood sugar level - high or very high (>140 mg/dl) or taking medicine to control blood sugar level23 (%) | 0.0241 |
| 13 | Population below age 15 years (%) | 0.0233 |
| 14 | Population living in households with electricity (%) | 0.0230 |
| 15 | Women with 10 or more years of schooling (%) | 0.0210 |

## Notes
- The test set has only 69 districts, so differences of a few points can be noise. Compare the repeated runs above.
- Feature importance shows association, not cause.
- Data: NFHS-5 district fact sheets (IIPS), compiled as CSV by Bhanu Pratap Vardhan, CC BY 4.0, https://doi.org/10.7910/DVN/42WNZF

