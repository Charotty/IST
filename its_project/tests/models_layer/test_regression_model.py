"""Tests for RegressionModel."""
import pytest
import numpy as np
from its_project.models.regression_model import RegressionModel


@pytest.fixture
def regression_config():
    """Regression model configuration."""
    return {
        "model_type": "linear",
    }


@pytest.fixture
def gradient_boosting_config():
    """Gradient boosting model configuration."""
    return {
        "model_type": "gradient_boosting",
        "n_estimators": 10,
        "learning_rate": 0.1,
        "max_depth": 2,
        "random_state": 42,
    }


@pytest.fixture
def sample_data():
    """Sample training and test data."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    X_test = np.random.randn(20, 10)
    y_test = np.random.randn(20)
    return X_train, y_train, X_test, y_test


def test_regression_model_init(regression_config):
    """Test regression model initialization."""
    model = RegressionModel(regression_config)
    assert model.model_type == "linear"
    assert not model._is_fitted


def test_gradient_boosting_model_init(gradient_boosting_config):
    """Test gradient boosting model initialization."""
    model = RegressionModel(gradient_boosting_config)
    assert model.model_type == "gradient_boosting"
    assert not model._is_fitted


def test_fit(regression_config, sample_data):
    """Test model fitting."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(regression_config)
    
    fitted_model = model.fit(X_train, y_train)
    
    assert fitted_model is model  # Returns self
    assert model._is_fitted


def test_predict(regression_config, sample_data):
    """Test model prediction."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(regression_config)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape == (len(X_test),)
    assert isinstance(predictions, np.ndarray)


def test_predict_proba(regression_config, sample_data):
    """Test probability prediction."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(regression_config)
    model.fit(X_train, y_train)
    
    proba = model.predict_proba(X_test)
    
    assert proba.shape == (len(X_test), 3)  # 3 classes: SELL, HOLD, BUY
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)  # Probabilities sum to 1


def test_predict_before_fit_raises_error(regression_config, sample_data):
    """Test that predict raises error when model not fitted."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(regression_config)
    
    with pytest.raises(RuntimeError, match="Model not fitted"):
        model.predict(X_test)


def test_gradient_boosting_fit(gradient_boosting_config, sample_data):
    """Test gradient boosting model fitting."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(gradient_boosting_config)
    
    model.fit(X_train, y_train)
    
    assert model._is_fitted
    assert hasattr(model.model, 'feature_importances_')


def test_get_feature_importance(gradient_boosting_config, sample_data):
    """Test feature importance retrieval."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(gradient_boosting_config)
    model.fit(X_train, y_train)
    
    importance = model.get_feature_importance()
    
    assert importance is not None
    assert len(importance) == X_train.shape[1]


def test_cross_validate(regression_config, sample_data):
    """Test cross-validation."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(regression_config)
    
    cv_results = model.cross_validate(X_train, y_train, cv=3)
    
    assert "cv_r2_mean" in cv_results
    assert "cv_r2_std" in cv_results
    assert isinstance(cv_results["cv_r2_mean"], float)


def test_3d_input_handling(regression_config, sample_data):
    """Test handling of 3D input (neural network style)."""
    X_train, y_train, X_test, y_test = sample_data
    
    # Convert to 3D
    X_train_3d = X_train.reshape(100, 1, 10)
    X_test_3d = X_test.reshape(20, 1, 10)
    
    model = RegressionModel(regression_config)
    model.fit(X_train_3d, y_train)
    
    predictions = model.predict(X_test_3d)
    
    assert predictions.shape == (len(X_test),)


def test_unknown_model_type():
    """Test that unknown model type raises error."""
    config = {"model_type": "unknown"}
    
    with pytest.raises(ValueError, match="Unknown model type"):
        RegressionModel(config)


def test_save_load(regression_config, sample_data, tmp_path):
    """Test model save and load."""
    X_train, y_train, X_test, y_test = sample_data
    model = RegressionModel(regression_config)
    model.fit(X_train, y_train)
    
    # Get predictions before save
    predictions_before = model.predict(X_test)
    
    # Save model
    model_path = tmp_path / "regression_model.joblib"
    model.save(model_path)
    
    # Load model
    loaded_model = RegressionModel.load(model_path)
    
    # Check loaded model
    assert loaded_model._is_fitted
    assert loaded_model.model_type == model.model_type
    
    # Check predictions match
    predictions_after = loaded_model.predict(X_test)
    np.testing.assert_array_almost_equal(predictions_before, predictions_after)
