import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ClassicTrAdaBoost:
    # learning_rate=0.1
    def __init__(self, base_learner_type='bpnn', n_estimators=30, learning_rate=0.1):
        self.base_learner_type = base_learner_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate  
        self.models = []
        self.alphas = []
        self.history_error = []

    def fit(self, X_source, y_source, X_target, y_target):
        ns, nt = X_source.shape[0], X_target.shape[0]
        X_train = np.concatenate((X_source, X_target), axis=0)
        y_train = np.concatenate((y_source, y_target), axis=0)
        
        weights = np.ones(ns + nt)
        weights[:ns] = (1.0 / ns) * 0.5
        weights[ns:] = (1.0 / nt) * 0.5
        weights /= np.sum(weights)
        beta_out = 1 / (1 + np.sqrt(2 * np.log(ns + 1) / self.n_estimators))
        
        for t in range(self.n_estimators):
            current_weights = weights / (np.sum(weights) + 1e-12)
            
            if self.base_learner_type == 'bpnn':
                indices = np.random.choice(len(X_train), size=len(X_train), p=current_weights)
                X_batch = X_train[indices]
                y_batch = y_train[indices]
                model = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500, random_state=t)
                model.fit(X_batch, y_batch)
            
            y_pred = model.predict(X_train)
            
            target_err = np.sum(current_weights[ns:] * (y_pred[ns:] != y_target)) / (np.sum(current_weights[ns:]) + 1e-12)
            self.history_error.append(target_err)
            
            if target_err >= 0.5:
                print(f"[{self.base_learner_type}] stop iteration")
                break
            
            target_err = max(target_err, 1e-10)
            beta_t = target_err / (1 - target_err)
            
       
            alpha = 0.5 * np.log(1 / beta_t) * self.learning_rate

            # updating weights
            new_weights = weights.copy()
            for i in range(len(weights)):
                if i < ns: # source domain
                    if y_pred[i] != y_train[i]: 
                        new_weights[i] = weights[i] * (beta_out)
                else: # target domain
                    if y_pred[i] != y_target[i-ns]: 
                        new_weights[i] = weights[i] * (1/beta_t)
            
            weights = new_weights
            self.models.append(model)
            self.alphas.append(alpha)

    def predict_proba(self, X):
        if not self.models: return np.ones(X.shape[0]) * 0.5
        final_proba = np.zeros(X.shape[0])
        total_alpha = np.sum(self.alphas) + 1e-12
        for i in range(len(self.models)):
            final_proba += (self.alphas[i] / total_alpha) * self.models[i].predict_proba(X)[:, 1]
        return final_proba


# grid research
def grid_search_tradaboost_lr(X_s, y_s, X_t, y_t, base_learner='bpnn'):
    """
    Scale: 0.0 to 1.0，with interval value of 0.05
    """
  
    lr_grid = np.round(np.arange(0.0, 1.05, 0.05), 2)
    
 
    X_t_tr, X_t_val, y_t_tr, y_t_val = train_test_split(
        X_t, y_t, test_size=0.3, random_state=42
    )
    
    best_lr = 0.1
    best_score = -1.0
    results = []

    print("================  ClassicTrAdaBoost.learning_rate ================")
    
    for lr in lr_grid:
        # 
        model = ClassicTrAdaBoost(base_learner_type=base_learner, n_estimators=30, learning_rate=lr)
        model.fit(X_s, y_s, X_t_tr, y_t_tr)
        
        # validation
        val_proba = model.predict_proba(X_t_val)
        val_auc = roc_auc_score(y_t_val, val_proba)
        
        results.append({'TrAdaBoost_learning_rate': lr, 'Validation_AUC': val_auc})
        
        if val_auc > best_score:
            best_score = val_auc
            best_lr = lr

    print("==========================================================================")
    print(f" Optimal TrAdaBoost learning_rate is: {best_lr} (AUC: {best_score:.4f})")
    
   
    best_model = ClassicTrAdaBoost(base_learner_type=base_learner, n_estimators=30, learning_rate=best_lr)
    best_model.fit(X_s, y_s, X_t, y_t)
    
    return best_model, best_lr, pd.DataFrame(results)