```python
# simplified_model_with_tuning.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

# ---------------------------------------------------------------------------
# 1. Data Loading and Preprocessing
# ---------------------------------------------------------------------------

raw_path="add.raw"
pheno_path="pheno.txt"

def load_genotype(raw_path, id_cols=["FID", "IID", "PAT", "MAT", "SEX"]):
    """
    Load PLINK .raw format genotype file and return DataFrame indexed by sample IID.
    """
    geno = pd.read_csv(raw_path, delim_whitespace=True)
    # Keep IID for merge, drop pedigree columns
    ids = geno[["IID"]]
    X = geno.drop(columns=id_cols)
    X.index = ids["IID"]
    return X


def load_phenotype(pheno_path, id_col="IID", pheno_col="PHENOTYPE"):  
    """
    Load phenotype file and return Series of phenotypes indexed by sample IID.
    """
    pheno = pd.read_csv(pheno_path)
    pheno = pheno.set_index(id_col)[pheno_col]
    return pheno


def filter_by_maf(X, low=0.005, high=0.995):
    """
    Filter genotype markers by minor allele frequency thresholds.
    Assumes genotypes coded as 0,1,2.
    """
    maf = X.mean(axis=0) / 2
    keep = maf[(maf >= low) & (maf <= high)].index
    return X[keep]

# ---------------------------------------------------------------------------
# 2. Model Definition and Tuning
# ---------------------------------------------------------------------------
def build_model(hp, input_dim):
    """
    Build a simple NN with hyperparameters chosen by Keras Tuner.
    """
    model = keras.Sequential()
    # Hidden layer
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    activation = hp.Choice('activation', ['relu', 'tanh'])
    l2_rate = hp.Float("L2", 0.01, 20, step=1.4, sampling = "log")
    model.add(
        layers.Dense(units,
                     activation=activation,
                     kernel_regularizer=regularizers.l2(l2_rate),
                     input_shape=(input_dim,))
    )
    # Dropout
    dropout = hp.Choice("dropout", values=[0.0, 0.01, 0.02,0.04,0.06,0.08, 0.1, 0.2, 0.3, 0.4, 0.5])
    model.add(layers.Dropout(dropout))
    # Output layer
    model.add(layers.Dense(1))

    # Compile
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    return model


def tune_hyperparameters(X_train, y_train, input_dim,
                         max_epochs=50,
                         factor=3,
                         project_name='tuner'):
    """
    Perform a Hyperband search to find optimal hyperparameters.
    Returns the best HyperParameters and the tuner object.
    """
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_dim),
        objective='val_mean_squared_error',
        max_epochs=max_epochs,
        factor=factor,
        directory='tuner_dir',
        project_name=project_name,
        overwrite=True
    )
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=max_epochs,
        callbacks=[stop_early],
        verbose=1
    )
    best_hps = tuner.get_best_hyperparameters(1)[0]
    return best_hps, tuner

# ---------------------------------------------------------------------------
# 3. Training, Evaluation, and Prediction
# ---------------------------------------------------------------------------
def train_and_evaluate(best_hps, tuner, X_train, y_train, X_test, y_test, batch_size=1024):
    """
    Build the model with best hyperparameters, train to best epoch, evaluate on test set.
    Returns trained model and predictions.
    """
    # Determine best epoch from tuning history
    hist = tuner.get_best_models(num_models=1)[0].fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=batch_size,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)],
        verbose=0
    )
    val_mse = hist.history['val_loss']
    best_epoch = int(np.argmin(val_mse)) + 1

    # Retrain on full training data
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        X_train, y_train,
        epochs=best_epoch,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)
    preds = model.predict(X_test).flatten()
    return model, eval_metrics, preds

# ---------------------------------------------------------------------------
# 4. Main Workflow
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # File paths (update accordingly)
    raw_geno_file = '/path/to/genotypes.raw'
    pheno_file    = '/path/to/phenotypes.csv'

    # 1) Load data
    X = load_genotype(raw_geno_file)
    y = load_phenotype(pheno_file)

    # 2) Align samples and filter markers
    df = X.join(y, how='inner')
    y = df['PHENOTYPE']
    X = df.drop(columns=['PHENOTYPE'])
    X = filter_by_maf(X, low=0.005, high=0.995)

    # 3) Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train = (y_train - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    # 4) Hyperparameter tuning
    best_hps, tuner = tune_hyperparameters(
        X_train, y_train,
        input_dim=X_train.shape[1],
        max_epochs=50,
        factor=3,
        project_name='genetic_pred'
    )
    print('Best hyperparameters:', best_hps.values)

    # 5) Train best model and predict
    model, metrics, preds = train_and_evaluate(
        best_hps, tuner, X_train, y_train, X_test, y_test
    )
    test_mse = metrics[0]
    test_corr = np.corrcoef(y_test, preds)[0,1]

    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test correlation: {test_corr:.4f}")
```
