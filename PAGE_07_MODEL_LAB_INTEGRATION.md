# PAGE 7: 🔬 MODEL LABORATORY
**Complete Integration Guide for Cursor AI**

---

## 📋 OVERVIEW

**Target File:** `pages/7_Model_Lab.py`  
**Merges:** Model_Lab.py + 6_Model_Optimization.py + Model_Performance_Dashboard.py + 7_Optimizer.py + 8_Explainability.py  
**Tabs:** 7 tabs  
**Estimated Time:** 12-15 hours  
**Priority:** MEDIUM

### Features Preserved:
✅ Model training (all model types)  
✅ Hyperparameter tuning (multiple methods)  
✅ Model performance tracking  
✅ Feature importance analysis  
✅ SHAP values  
✅ Model versioning  
✅ AutoML capabilities

---

## CURSOR PROMPT 7.1 - Create Page Structure

```
Create pages/7_Model_Lab.py with 7-tab structure

BACKEND IMPORTS:
from trading.models.* import *
from trading.optimization.optuna_tuner import OptunaTuner
from trading.evaluation.model_evaluator import ModelEvaluator
from trading.analytics.forecast_explainability import ForecastExplainability
from trading.integration.model_registry import ModelRegistry

TABS:
1. Quick Training
2. Model Configuration
3. Hyperparameter Optimization
4. Model Performance
5. Model Comparison
6. Explainability
7. Model Registry

Most complex page - take time with this one.
```

---

## CURSOR PROMPT 7.2 - Implement Quick Training (Tab 1)

```
Implement Tab 1 (Quick Training) in pages/7_Model_Lab.py

Simple interface for fast model training:
- Data upload or select from history
- Model type dropdown (LSTM, XGBoost, Prophet, ARIMA, etc.)
- Train button
- Training progress bar
- Quick results display (accuracy, loss)
- Save model button

Default parameters, minimal configuration.
```

---

## CURSOR PROMPT 7.3 - Implement Model Configuration (Tab 2)

```
Implement Tab 2 (Model Configuration) in pages/7_Model_Lab.py

Detailed model configuration:
- Model architecture selection
- Layer-by-layer configuration (for neural networks)
- All hyperparameters exposed
- Feature engineering pipeline builder
- Preprocessing configuration
- Train/validation split settings
- Cross-validation options
- Training with monitoring

More advanced than Tab 1.
```

---

## CURSOR PROMPT 7.4 - Implement Hyperparameter Optimization (Tab 3)

```
Implement Tab 3 (Hyperparameter Optimization) in pages/7_Model_Lab.py

Automated hyperparameter tuning:
- Optimization method selector:
  * Grid Search
  * Random Search
  * Bayesian Optimization (Optuna)
  * Genetic Algorithm
- Search space configuration
- Optimization objective (minimize loss, maximize accuracy, etc.)
- Number of trials
- Run optimization button
- Live optimization progress:
  * Current trial
  * Best params so far
  * Optimization history plot
- Results table with all trials
- Best parameters export

Use Optuna for Bayesian optimization.
```

---

## CURSOR PROMPT 7.5 - Implement Model Performance (Tab 4)

```
Implement Tab 4 (Model Performance) in pages/7_Model_Lab.py

Comprehensive performance tracking:
- Training metrics:
  * Loss curves (train/val)
  * Accuracy curves
  * Epoch-by-epoch metrics
- Validation metrics:
  * Confusion matrix (classification)
  * Residual plots (regression)
  * ROC curves (classification)
  * Prediction vs actual scatter
- Test set evaluation
- Performance over time tracking
- Model degradation detection

Save all metrics to database.
```

---

## CURSOR PROMPT 7.6 - Implement Model Comparison (Tab 5)

```
Implement Tab 5 (Model Comparison) in pages/7_Model_Lab.py

Side-by-side model comparison:
- Select multiple trained models
- Comparison metrics table
- Performance charts overlay
- Statistical significance tests
- Model selection recommendation
- Ensemble creation option

Help user choose best model.
```

---

## CURSOR PROMPT 7.7 - Implement Explainability (Tab 6)

```
Implement Tab 6 (Explainability) in pages/7_Model_Lab.py

Model interpretability tools:
- Feature importance (all methods)
- SHAP values:
  * Summary plot
  * Dependence plots
  * Force plots
  * Waterfall plots
- LIME explanations
- Partial dependence plots
- Individual prediction explanations
- Model behavior analysis

Make black-box models interpretable.
```

---

## CURSOR PROMPT 7.8 - Implement Model Registry (Tab 7)

```
Implement Tab 7 (Model Registry) in pages/7_Model_Lab.py

Model version control and deployment:
- Saved models library (searchable table)
- Model metadata:
  * Name, version
  * Training date
  * Performance metrics
  * Parameters used
  * Data used
- Load model button
- Deploy to production
- Model comparison history
- Model lineage (parent models)
- Export/import models
- Model notes/tags

Full MLOps capabilities.
```

---

## ✅ PAGE 7 CHECKLIST

- [ ] File created: pages/7_Model_Lab.py
- [ ] All 7 tabs implemented
- [ ] Quick training works
- [ ] Full configuration available
- [ ] Hyperparameter optimization functional
- [ ] Performance tracking comprehensive
- [ ] Model comparison works
- [ ] Explainability tools functional
- [ ] Model registry operational
- [ ] All visualizations display
- [ ] Committed to git

---

## 🚀 COMMIT COMMAND

```bash
git add pages/7_Model_Lab.py
git commit -m "feat(page-7): Implement Model Laboratory with 7 tabs

- Tab 1: Quick training interface
- Tab 2: Detailed model configuration
- Tab 3: Automated hyperparameter optimization
- Tab 4: Comprehensive performance tracking
- Tab 5: Multi-model comparison
- Tab 6: Model explainability (SHAP, LIME)
- Tab 7: Model registry and versioning

Merges 5 model-related pages with full MLOps"
```

---

**Next:** PAGE_08_REPORTS_INTEGRATION.md
