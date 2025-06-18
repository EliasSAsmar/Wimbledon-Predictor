from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


class ServeReturnModelHelper:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def train(self, X_train, y_train):
        X = X_train[
            ['serve_rating_diff', 'serve_rating_grass_diff', 'return_rating_diff', 'return_rating_grass_diff']
        ]
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y_train)

    def predict_proba(self, X_test):
        X = X_test[
            ['serve_rating_diff', 'serve_rating_grass_diff', 'return_rating_diff', 'return_rating_grass_diff']
        ]
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X_test):
        X = X_test[
            ['serve_rating_diff', 'serve_rating_grass_diff', 'return_rating_diff', 'return_rating_grass_diff']
        ]
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        probs = self.predict_proba(X_test)
        preds = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, preds),
            'auc_roc': roc_auc_score(y_test, probs),
            'log_loss': log_loss(y_test, probs)
        }
