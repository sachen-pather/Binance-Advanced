"""
Advanced Machine Learning Engine with XGBoost, LightGBM, and Ensemble Methods
for sophisticated trading signal generation and market prediction.
"""

import pandas as pd
import numpy as np
import pickle
import os
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import optuna
from optuna.samplers import TPESampler

# Advanced ML libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost and/or LightGBM not available. Install with: pip install xgboost lightgbm")
    ADVANCED_ML_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from scikeras.wrappers import KerasClassifier
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. Neural network models will be disabled.")
    TENSORFLOW_AVAILABLE = False

from config import STRATEGY_PARAMS, ENV_CONFIG

logger = logging.getLogger("BinanceTrading.AdvancedML")
warnings.filterwarnings('ignore', category=UserWarning)


def create_neural_network(input_dim: int) -> Any:
    """Create neural network model for ensemble"""
    # NOTE: You will need to import Sequential, Dense, etc. inside this function
    # or ensure they are available in the global scope.
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        # Assuming logger is configured globally or passed in
        # logger.error(f"Error creating neural network: {e}")
        print(f"Error creating neural network: {e}") # Simple print for now
        return None

class AdvancedMLEngine:
    """Advanced machine learning engine with multiple algorithms and ensemble methods"""
    
    def __init__(self, strategy_params=None):
        self.strategy_params = strategy_params or STRATEGY_PARAMS
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.model_performance = {}
        
        # Model configurations
        self.model_configs = {
            'xgb': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lgb': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1
            },
            'rf': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        }
        
    def create_advanced_features(self, df: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Create advanced features for ML models including lag features and rolling statistics"""
        
        try:
            features_df = pd.DataFrame(index=df.index)
            
            # Price-based features
            features_df['returns'] = df['close'].pct_change()
            features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility features
            for period in lookback_periods:
                features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
                features_df[f'realized_vol_{period}'] = (features_df['log_returns'] ** 2).rolling(period).sum()
            
            # Technical indicator features (normalized)
            technical_indicators = [
                'RSI_14', 'RSI_7', 'MACD', 'MACD_signal', 'BB_position', 'ADX', 
                'STOCH_K', 'STOCH_D', 'CCI', 'MFI', 'WILLR', 'ROC_10', 'ATR_14'
            ]
            
            for indicator in technical_indicators:
                if indicator in df.columns:
                    # Normalize indicators
                    features_df[f'{indicator}_norm'] = self._robust_normalize(df[indicator])
                    
                    # Add momentum features
                    features_df[f'{indicator}_momentum'] = df[indicator].pct_change()
                    
                    # Add cross-sectional features
                    for period in [5, 10, 20]:
                        features_df[f'{indicator}_ma_{period}'] = df[indicator].rolling(period).mean()
                        features_df[f'{indicator}_std_{period}'] = df[indicator].rolling(period).std()
                        features_df[f'{indicator}_zscore_{period}'] = (df[indicator] - features_df[f'{indicator}_ma_{period}']) / features_df[f'{indicator}_std_{period}']
            
            # Volume features
            if 'volume' in df.columns:
                features_df['volume_norm'] = self._robust_normalize(df['volume'])
                features_df['volume_momentum'] = df['volume'].pct_change()
                
                for period in lookback_periods:
                    features_df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
                    features_df[f'volume_zscore_{period}'] = (df['volume'] - df['volume'].rolling(period).mean()) / df['volume'].rolling(period).std()
            
            # Price patterns
            for period in lookback_periods:
                features_df[f'high_low_ratio_{period}'] = (df['high'] - df['low']) / df['close']
                features_df[f'close_position_{period}'] = (df['close'] - df['low'].rolling(period).min()) / (df['high'].rolling(period).max() - df['low'].rolling(period).min())
            
            # Time-based features
            features_df['hour'] = df.index.hour / 23.0
            features_df['day_of_week'] = df.index.dayofweek / 6.0
            features_df['month'] = df.index.month / 11.0
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
                if 'RSI_14_norm' in features_df.columns:
                    features_df[f'rsi_lag_{lag}'] = features_df['RSI_14_norm'].shift(lag)
            
            # Rolling correlation features
            if len(features_df) > 50:
                features_df['returns_autocorr_5'] = features_df['returns'].rolling(20).apply(
                    lambda x: x.autocorr(lag=5) if len(x) >= 10 else 0
                )
                features_df['returns_autocorr_10'] = features_df['returns'].rolling(20).apply(
                    lambda x: x.autocorr(lag=10) if len(x) >= 15 else 0
                )
            
            # Market microstructure features
            if 'Amihud_Illiq' in df.columns:
                features_df['illiquidity'] = self._robust_normalize(df['Amihud_Illiq'])
            
            if 'VPIN' in df.columns:
                features_df['vpin'] = df['VPIN'].fillna(0.5)
            
            # Clean features
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            logger.info(f"Created {len(features_df.columns)} advanced features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating advanced features: {e}")
            return pd.DataFrame()
    
    def _robust_normalize(self, series: pd.Series) -> pd.Series:
        """Robust normalization using median and MAD"""
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return series / series.std() if series.std() != 0 else series
        return (series - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal distribution
    
    def create_labels(self, df: pd.DataFrame, method: str = 'adaptive', forward_periods: int = 24) -> np.array:
        """Create sophisticated labels for ML training"""
        
        try:
            if method == 'adaptive':
                # Adaptive labeling based on ATR
                atr = df.get('ATR_14', df['close'].rolling(14).std())
                threshold = atr.rolling(50).median() / df['close']
                
                forward_returns = df['close'].shift(-forward_periods) / df['close'] - 1
                
                # Dynamic thresholds
                upper_threshold = threshold * 1.5
                lower_threshold = -threshold * 1.0
                
                labels = np.where(forward_returns > upper_threshold, 1,  # Strong buy
                         np.where(forward_returns < lower_threshold, -1,  # Strong sell
                         0))  # Hold
                
            elif method == 'trend_following':
                # Trend-following labels
                future_trend = df['close'].shift(-forward_periods) / df['close'].shift(-1) - 1
                trend_threshold = df['close'].rolling(20).std() / df['close']
                
                labels = np.where(future_trend > trend_threshold, 1,
                         np.where(future_trend < -trend_threshold, -1, 0))
                
            elif method == 'mean_reversion':
                # Mean reversion labels
                price_zscore = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
                future_zscore = price_zscore.shift(-forward_periods)
                
                labels = np.where((price_zscore < -1.5) & (future_zscore > -0.5), 1,  # Buy oversold
                         np.where((price_zscore > 1.5) & (future_zscore < 0.5), -1,   # Sell overbought
                         0))
                
            else:  # 'simple'
                forward_returns = df['close'].shift(-forward_periods) / df['close'] - 1
                threshold = 0.02  # 2% threshold
                
                labels = np.where(forward_returns > threshold, 1,
                         np.where(forward_returns < -threshold, -1, 0))
            
            return labels[:len(df)]
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return np.zeros(len(df))
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: np.array, model_type: str = 'xgb', n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna"""

        if not ADVANCED_ML_AVAILABLE and model_type in ['xgb', 'lgb']:
            logger.warning(f"Advanced ML libraries not available for {model_type}")
            return self.model_configs.get(model_type, {})
        
        def objective(trial):
            if model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    
                    # <<< --- GPU ACTIVATION FOR OPTUNA (XGBoost) --- >>>
                    'tree_method': 'hist',
                    'device': 'cuda'
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_type == 'lgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'verbose': -1,
                    'random_state': 42,

                    # <<< --- GPU ACTIVATION FOR OPTUNA (LightGBM) --- >>>
                }
                model = lgb.LGBMClassifier(**params)
                
            else:  # Random Forest (CPU-only, no change needed)
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            # <<< --- CRITICAL FIX: Set n_jobs to 1 to prevent GPU conflicts --- >>>
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=1)
            return scores.mean()
        
        try:
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(objective, n_trials=n_trials, timeout=300)  # 5 minute timeout
            
            logger.info(f"Best {model_type} parameters: {study.best_params}")
            logger.info(f"Best {model_type} score: {study.best_value:.4f}")
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Error optimizing {model_type} hyperparameters: {e}")
            return self.model_configs.get(model_type, {})
    
    def train_ensemble_model(self, data_fetcher, indicator_calculator, symbols: List[str] = ['BTCUSDT', 'ETHUSDT'], 
                           optimize_hyperparams: bool = True) -> Tuple[bool, float]:
        """Train ensemble model with multiple algorithms"""
        
        try:
            all_features = []
            all_labels = []
            
            # Collect training data
            for symbol in symbols:
                logger.info(f"Collecting training data for {symbol}")
                
                df = data_fetcher.get_historical_data(symbol, '1h', lookback='90 days')
                if df is None or len(df) < 200:
                    continue
                
                # Calculate enhanced indicators
                df = indicator_calculator.calculate_all_indicators(df)
                if df is None:
                    continue
                
                # Create advanced features
                features = self.create_advanced_features(df)
                if features.empty:
                    continue
                
                # Create labels using multiple methods and combine
                labels_adaptive = self.create_labels(df, method='adaptive')
                labels_trend = self.create_labels(df, method='trend_following') 
                labels_reversion = self.create_labels(df, method='mean_reversion')
                
                # Combine labels with voting
                combined_labels = []
                for i in range(len(labels_adaptive)):
                    votes = [labels_adaptive[i], labels_trend[i], labels_reversion[i]]
                    # Take majority vote, default to 0 if tie
                    label_counts = {-1: 0, 0: 0, 1: 0}
                    for vote in votes:
                        label_counts[vote] += 1
                    combined_labels.append(max(label_counts, key=label_counts.get))
                
                labels = np.array(combined_labels)
                
                # Remove NaN values
                valid_indices = ~(features.isna().any(axis=1) | np.isnan(labels))
                features_clean = features[valid_indices]
                labels_clean = labels[valid_indices]
                
                if len(features_clean) > 50:
                    all_features.append(features_clean)
                    all_labels.extend(labels_clean)
            
            if not all_features:
                logger.error("No valid training data collected")
                return False, 0.0
            
            # CORRECTED CODE
            # Combine all features
            X = pd.concat(all_features, ignore_index=True)
             # Save the column order BEFORE any feature selection
            self.training_columns_ = X.columns.tolist()
            logger.info(f"Saved {len(self.training_columns_)} training column names for prediction alignment.")
            y_original = np.array(all_labels) # Keep original for reference if needed
        
            
            # --- FIX: Map labels from [-1, 0, 1] to [0, 1, 2] ---
            y = y_original + 1
            # Check the unique values to be sure: should be {0, 1, 2}
            logger.info(f"Original labels unique values: {np.unique(y_original)}. Mapped to: {np.unique(y)}")
            # -----------------------------------------------------

            logger.info(f"Training ensemble with {len(X)} samples and {len(X.columns)} features")

            # Feature selection (uses y)
            self.feature_selectors['selectk'] = SelectKBest(f_classif, k=min(50, len(X.columns)))
            X_selected = self.feature_selectors['selectk'].fit_transform(X, y)

            # Scale features
            self.scalers['robust'] = RobustScaler()
            X_scaled = self.scalers['robust'].fit_transform(X_selected)

            # Initialize models
            base_models = []
            
            # Random Forest (always available)
            if optimize_hyperparams:
                rf_params = self.optimize_hyperparameters(pd.DataFrame(X_scaled), y, 'rf')
            else:
                rf_params = self.model_configs['rf']
            
            rf_model = RandomForestClassifier(**rf_params, random_state=42)
            base_models.append(('rf', rf_model))
            
            # XGBoost (if available)
            if ADVANCED_ML_AVAILABLE:
                if optimize_hyperparams:
                    xgb_params = self.optimize_hyperparameters(pd.DataFrame(X_scaled), y, 'xgb')
                else:
                    xgb_params = self.model_configs['xgb']
                
                # <<< --- GPU ACTIVATION for FINAL XGBoost MODEL --- >>>
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['device'] = 'cuda'
                
                xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
                base_models.append(('xgb', xgb_model))
                
                # LightGBM
                if optimize_hyperparams:
                    lgb_params = self.optimize_hyperparameters(pd.DataFrame(X_scaled), y, 'lgb')
                else:
                    lgb_params = self.model_configs['lgb']
                
                # <<< --- GPU ACTIVATION for FINAL LightGBM MODEL --- >>>
                #lgb_params['device'] = 'gpu'
                
                lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
                base_models.append(('lgb', lgb_model))
            
            # Neural Network (if TensorFlow available)
           # if TENSORFLOW_AVAILABLE and len(X_scaled) > 1000:
            # Create a wrapper function that returns the model
            #    def create_nn_wrapper():
              #      return create_neural_network(X_scaled.shape[1])
                
            #    nn_model = KerasClassifier(
              #      model=create_nn_wrapper,
              #     epochs=50,
              #      batch_size=128,
               #     verbose=0,
               #     callbacks=[EarlyStopping(patience=5, monitor='val_loss', mode='min')]
               # )
               # base_models.append(('nn', nn_model))
            
            # Create ensemble using stacking
            if len(base_models) > 1:
                meta_learner = RandomForestClassifier(n_estimators=50, random_state=42)
                self.ensemble_model = StackingClassifier(
                    estimators=base_models,
                    final_estimator=meta_learner,
                    cv=KFold(n_splits=3), 
                    stack_method='predict_proba',
                    n_jobs=1,
                    passthrough=True
                )
            else:
                # Fallback to single model if only one available
                self.ensemble_model = base_models[0][1]
            
            # Train ensemble
            logger.info("Training ensemble model...")
            self.ensemble_model.fit(X_scaled, y)
            
            # Evaluate performance
 # --- START OF FIX: Evaluate performance using PROPER cross-validation ---
            logger.info("Evaluating final ensemble model using cross-validation...")
            
            # Use TimeSeriesSplit for a more robust evaluation on time-series data
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Calculate cross-validated scores
            # n_jobs=1 is important, especially with GPU, to prevent conflicts.
            cv_scores = cross_val_score(self.ensemble_model, X_scaled, y, cv=tscv, scoring='accuracy', n_jobs=1)
            
            accuracy = cv_scores.mean()
            cv_std = cv_scores.std()

            # For other metrics, we can't easily get them from cross_val_score.
            # A common practice is to use a single train-test split for a representative metric calculation.
            # Let's use the last fold of TimeSeriesSplit for this.
            train_index, test_index = list(tscv.split(X_scaled))[-1]
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # We need to re-fit a temporary model on just the training part of the split
            # to make predictions on the test part.
            temp_model = self.ensemble_model # You can clone it if you want a clean object
            temp_model.fit(X_train, y_train)
            y_pred_test = temp_model.predict(X_test)

            precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            
            self.model_performance = {
                'accuracy': accuracy,
                'cv_std': cv_std, # Standard deviation of accuracy across folds
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_samples': len(X_scaled)
            }
            
            # Feature importance (if available)
            try:
                if hasattr(self.ensemble_model, 'feature_importances_'):
                    selected_features = self.feature_selectors['selectk'].get_support()
                    feature_names = np.array(X.columns)[selected_features]
                    importances = self.ensemble_model.feature_importances_
                    
                    self.feature_importance = dict(zip(feature_names, importances))
                    
                    # Log top features
                    top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    logger.info("Top 10 features:")
                    for feat, imp in top_features:
                        logger.info(f"  {feat}: {imp:.4f}")
                        
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {e}")
            
            # Save models
            self.save_models()
            
            # Update strategy parameters
            self.strategy_params['last_retrain_time'] = datetime.now()
            
            logger.info(f"Ensemble model training completed:")
            logger.info(f"  Accuracy: {accuracy:.4f} ± {cv_std:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            
            return True, accuracy
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, 0.0
    
    def predict_ensemble(self, features: pd.DataFrame) -> Tuple[int, float]:
        """Make predictions using ensemble model"""
        
        if self.ensemble_model is None:
            logger.warning("No ensemble model available for prediction")
            return 0, 0.0
        
        try:
            # Prepare features - ensure we have the same columns as training
            if not hasattr(self, 'training_columns_'):
                logger.error("Model has not been trained yet, training_columns_ attribute is missing.")
                return 0, 0.0
            
            # Align features to match training columns exactly
            aligned_features = features.reindex(columns=self.training_columns_, fill_value=0)
            
            # Convert to numpy array and ensure 2D shape (1 sample, n_features)
            X_input = aligned_features.values
            if X_input.ndim == 1:
                X_input = X_input.reshape(1, -1)
            
            # Apply feature selection if available
            if 'selectk' in self.feature_selectors:
                X_selected = self.feature_selectors['selectk'].transform(X_input)
            else:
                X_selected = X_input
            
            # Apply scaling if available
            if 'robust' in self.scalers:
                X_scaled = self.scalers['robust'].transform(X_selected)
            else:
                X_scaled = X_selected
            
            # Ensure X_scaled is 2D
            if X_scaled.ndim == 1:
                X_scaled = X_scaled.reshape(1, -1)
            
            # Make prediction
            raw_prediction = self.ensemble_model.predict(X_scaled)[0]

            # Map prediction back from [0, 1, 2] to [-1, 0, 1]
            prediction = raw_prediction - 1

            # Get prediction probability/confidence
            if hasattr(self.ensemble_model, 'predict_proba'):
                probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.7  # Default confidence

            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return 0, 0.0
    
    def save_models(self):
        """Save trained models and preprocessors"""
        
        try:
            model_data = {
                'ensemble_model': self.ensemble_model,
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open(ENV_CONFIG['model_file'], 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Advanced ML models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
# In advanced_ml_engine.py -> class AdvancedMLEngine

    def save_models(self):
        """Save trained models and preprocessors"""
        
        try:
            model_data = {
                'ensemble_model': self.ensemble_model,
                'scalers': self.scalers,
                'feature_selectors': self.feature_selectors,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'training_timestamp': datetime.now().isoformat(),
                # --- THIS IS THE FIX ---
                'training_columns': self.training_columns_  # Add the column list to the saved data
                # --- END OF FIX ---
            }
            
            with open(ENV_CONFIG['model_file'], 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Advanced ML models saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load trained models and preprocessors"""
        
        try:
            if os.path.exists(ENV_CONFIG['model_file']):
                with open(ENV_CONFIG['model_file'], 'rb') as f:
                    model_data = pickle.load(f)
                
                self.ensemble_model = model_data.get('ensemble_model')
                self.scalers = model_data.get('scalers', {})
                self.feature_selectors = model_data.get('feature_selectors', {})
                self.feature_importance = model_data.get('feature_importance', {})
                self.model_performance = model_data.get('model_performance', {})
                # --- THIS IS THE FIX ---
                self.training_columns_ = model_data.get('training_columns', []) # Load the column list
                if not self.training_columns_:
                    logger.warning("Loaded model did not contain training columns. Predictions may fail if feature order changes.")
                else:
                    logger.info(f"Loaded {len(self.training_columns_)} training column names from model file.")
                # --- END OF FIX ---
                
                logger.info("Advanced ML models loaded successfully")
                return True
            else:
                logger.info("No saved models found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_feature_importance_analysis(self) -> Dict:
        """Get detailed feature importance analysis"""
        
        if not self.feature_importance:
            return {"error": "No feature importance data available"}
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize features
        categories = {
            'trend': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'statistical': [],
            'time': [],
            'other': []
        }
        
        for feature, importance in sorted_features:
            if any(x in feature.lower() for x in ['sma', 'ema', 'macd', 'adx', 'trend']):
                categories['trend'].append((feature, importance))
            elif any(x in feature.lower() for x in ['rsi', 'stoch', 'roc', 'mom', 'cci']):
                categories['momentum'].append((feature, importance))
            elif any(x in feature.lower() for x in ['atr', 'bb', 'volatility', 'std']):
                categories['volatility'].append((feature, importance))
            elif any(x in feature.lower() for x in ['volume', 'obv', 'vpin', 'mfi']):
                categories['volume'].append((feature, importance))
            elif any(x in feature.lower() for x in ['zscore', 'correlation', 'linreg', 'hurst']):
                categories['statistical'].append((feature, importance))
            elif any(x in feature.lower() for x in ['hour', 'day', 'month', 'time']):
                categories['time'].append((feature, importance))
            else:
                categories['other'].append((feature, importance))
        
        return {
            'top_features': sorted_features[:20],
            'categories': categories,
            'model_performance': self.model_performance
        }
    
    def retrain_if_needed(self, data_fetcher, indicator_calculator, force_retrain: bool = False) -> bool:
        """Retrain model if performance has degraded or time threshold reached"""
        
        try:
            # Check if retraining is needed
            if not force_retrain:
                last_retrain = self.strategy_params.get('last_retrain_time')
                if last_retrain:
                    if isinstance(last_retrain, str):
                        last_retrain = datetime.fromisoformat(last_retrain)
                    
                    hours_since_retrain = (datetime.now() - last_retrain).total_seconds() / 3600
                    if hours_since_retrain < self.strategy_params.get('ml_retrain_hours', 24):
                        logger.info(f"Model recently trained {hours_since_retrain:.1f} hours ago, skipping retrain")
                        return False
            
            logger.info("Retraining ensemble model...")
            success, accuracy = self.train_ensemble_model(
                data_fetcher, 
                indicator_calculator, 
                optimize_hyperparams=True
            )
            
            if success:
                logger.info(f"Model retrained successfully with accuracy: {accuracy:.4f}")
                return True
            else:
                logger.error("Model retraining failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            return False


class ModelPerformanceMonitor:
    """Monitor model performance and trigger retraining when needed"""
    
    def __init__(self):
        self.prediction_history = []
        self.performance_metrics = {}
        
    def log_prediction(self, prediction: int, confidence: float, actual_result: Optional[int] = None):
        """Log prediction and actual result for performance tracking"""
        
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'actual': actual_result
        })
        
        # Keep only recent predictions (last 1000)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def calculate_recent_performance(self, days: int = 7) -> Dict:
        """Calculate performance metrics for recent period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_predictions = [
            p for p in self.prediction_history 
            if p['timestamp'] > cutoff_date and p['actual'] is not None
        ]
        
        if len(recent_predictions) < 10:
            return {"insufficient_data": True}
        
        correct_predictions = sum(1 for p in recent_predictions if p['prediction'] == p['actual'])
        accuracy = correct_predictions / len(recent_predictions)
        
        avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(recent_predictions),
            'correct_predictions': correct_predictions,
            'average_confidence': avg_confidence,
            'period_days': days
        }
    
    def should_retrain(self, min_accuracy: float = 0.55, min_predictions: int = 50) -> bool:
        """Determine if model should be retrained based on recent performance"""
        
        recent_perf = self.calculate_recent_performance()
        
        if recent_perf.get('insufficient_data'):
            return False
        
        if recent_perf['total_predictions'] < min_predictions:
            return False
        
        if recent_perf['accuracy'] < min_accuracy:
            logger.warning(f"Model accuracy ({recent_perf['accuracy']:.3f}) below threshold ({min_accuracy})")
            return True
        
        return False