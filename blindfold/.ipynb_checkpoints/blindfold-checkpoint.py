import xgboost

class BlindFold(object):
    def __init__(self,
                alpha=555,
                booster='gbtree',
                colsample_bytree = 0.85,
                eta = 0.27,
                gamma = 5.16,
                reg_lambda = 945,
                max_depth = 6,
                min_child_weight=11,
                n_estimators= 10,
                objective='rank:pairwise',
                seed=2023,
                subsample=0.8,
                tree_method='hist'
                ):
                
        assert alpha >= 0.0
        assert colsample_bytree >= 0.0
        assert eta > 0
        assert max_depth > 0
        assert min_child_weight > 0
        assert 0 < subsample < 1
        assert n_estimators >= 1
        assert tree_method in ("hist")
        assert objective in ("rank:pairwise")

        self.alpha=alpha,
        self.booster=booster,
        self.colsample_bytree = colsample_bytree,
        self.eta = eta,
        self.gamma = gamma,
        self.reg_lambda = reg_lambda,
        self.max_depth = max_depth,
        self.min_child_weight=min_child_weight,
        self.n_estimators= n_estimators,
        self.objective=objective,
        self.seed=seed,
        self.subsample=subsample,
        self.tree_method=tree_method
        
        
    def fit(self, X_train, y_train, qids_train, X_test, y_test, qids_test):
        
        self.model = xgboost.XGBRanker(
                    alpha=self.alpha[0],
                    booster=self.booster[0],
                    colsample_bytree = self.colsample_bytree[0],
                    eta = self.eta[0],
                    gamma = self.gamma[0],
                    reg_lambda = self.reg_lambda[0],
                    max_depth = self.max_depth[0],
                    min_child_weight=self.min_child_weight[0],
                    n_estimators= self.n_estimators[0],
                    objective=self.objective[0],
                    seed=self.seed[0],
                    subsample=self.subsample[0],
                    )
        
        self.model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train),
                    (X_test, y_test)], 
          group=qids_train, 
          eval_group = [qids_train, qids_test], 
          eval_metric = ["ndcg@10"], 
          early_stopping_rounds=100,
          verbose=True)
        
        return self.model
