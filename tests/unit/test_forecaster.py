import numpy as np
import pandas as pd
import pytest

from global_forecasting import GlobalForecaster
from tests.fixtures.data_fixtures import sample_data
from tests.fixtures.model_fixtures import random_forest_model


class TestPredict:
    """Test suite for the predict method."""

    def test_predict_with_lags(self, sample_data, random_forest_model):
        """Test prediction with lag features."""
        # Initialize forecaster
        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1, 7])

        # Fit and predict
        forecaster.fit(sample_data)
        predictions = forecaster.predict(horizon=5)

        # Assertions
        assert len(predictions) == len(sample_data["unique_id"].unique()) * 5
        assert predictions["ds"].min() > sample_data["ds"].max()
        assert (
            (predictions.groupby("unique_id")["ds"].diff().dt.days == 1).dropna().all()
        )

    def test_predict_with_static_and_dynamic_features(
        self, sample_data, random_forest_model
    ):
        """Test prediction with both static and dynamic features."""
        # Initialize forecaster
        forecaster = GlobalForecaster(
            model=random_forest_model,
            freq="D",
            lags=[1, 7],
            static_features=["static_feature"],
            dynamic_features=["dynamic_feature"],
        )

        # Prepare future features
        future_dates = pd.date_range(
            start=sample_data["ds"].max() + pd.Timedelta(days=1), periods=5, freq="D"
        )

        future_features = []
        for sid in sample_data["unique_id"].unique():
            future_df = pd.DataFrame(
                {
                    "ds": future_dates,
                    "unique_id": sid,
                    "static_feature": sample_data[sample_data["unique_id"] == sid][
                        "static_feature"
                    ].iloc[0],
                    "dynamic_feature": np.random.normal(0, 1, len(future_dates)),
                }
            )
            future_features.append(future_df)

        future_features_df = pd.concat(future_features, ignore_index=True)

        # Fit and predict
        forecaster.fit(sample_data)
        predictions = forecaster.predict(horizon=5, X_df=future_features_df)

        # Assertions
        n_series = len(sample_data["unique_id"].unique())
        assert len(predictions) == n_series * 5
        assert set(predictions["unique_id"]) == set(sample_data["unique_id"].unique())
        assert all(predictions["ds"] >= sample_data["ds"].max())

    def test_predict_maintains_order(self, sample_data, random_forest_model):
        """Test that predictions maintain the correct order of series and dates."""
        # Initialize forecaster
        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1])

        # Fit and predict
        forecaster.fit(sample_data)
        predictions = forecaster.predict(horizon=3)

        # Check order
        series_order = sample_data["unique_id"].unique()
        pred_order = predictions.groupby("ds")["unique_id"].apply(list)

        # Assertions
        assert all(ids == series_order.tolist() for ids in pred_order)

    def test_predict_handles_missing_values(self, sample_data, random_forest_model):
        """Test that prediction handles missing values correctly."""
        # Create missing values
        data = sample_data.copy()
        data.loc[data.index[5:8], "y"] = np.nan

        # Initialize forecaster
        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1, 2])

        # Fit and predict
        forecaster.fit(data)
        predictions = forecaster.predict(horizon=3)

        # Assertions
        n_series = len(data["unique_id"].unique())
        assert len(predictions) == n_series * 3
        assert not predictions["y"].isna().any()

    def test_predict_raises_error_without_fit(self, sample_data, random_forest_model):
        """Test that prediction raises error if model is not fitted."""
        forecaster = GlobalForecaster(model=random_forest_model, freq="D")

        with pytest.raises(
            ValueError, match="Model must be fitted before making predictions"
        ):
            forecaster.predict(horizon=5)


class TestGenerateLags:
    """Test suite for the _generate_lags method."""

    def test_no_missing_lag_generation(
        self, sample_data_no_missing, random_forest_model
    ):
        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1, 2])
        result_df = forecaster._generate_lags(
            sample_data_no_missing.copy(),
            ds_col="ds",
            unique_id_col="unique_id",
            target_col="y",
        )

        assert "y_lag_1" in result_df.columns
        assert "y_lag_2" in result_df.columns

        # Check the values
        df_expected = sample_data_no_missing.copy()
        for lag in [1, 2]:
            temp = sample_data_no_missing.rename(columns={"y": f"y_lag_{lag}"})
            temp["ds"] = temp["ds"] + pd.to_timedelta(lag, unit="D")
            df_expected = pd.merge(
                df_expected,
                temp[["unique_id", "ds", f"y_lag_{lag}"]],
                on=["unique_id", "ds"],
                how="left",
            )

        # Check the values
        for lag in [1, 2]:
            pd.testing.assert_series_equal(
                result_df[f"y_lag_{lag}"], df_expected[f"y_lag_{lag}"]
            )

    def test_missing_data_lag_generation(
        self, sample_data_with_missing, random_forest_model
    ):
        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1, 2])
        result_df = forecaster._generate_lags(
            sample_data_with_missing.copy(),
            ds_col="ds",
            unique_id_col="unique_id",
            target_col="y",
        )

        assert "y_lag_1" in result_df.columns
        assert "y_lag_2" in result_df.columns

        # Check the values
        df_expected = sample_data_with_missing.copy()
        for lag in [1, 2]:
            temp = sample_data_with_missing.rename(columns={"y": f"y_lag_{lag}"})
            temp["ds"] = temp["ds"] + pd.to_timedelta(lag, unit="D")
            df_expected = pd.merge(
                df_expected,
                temp[["unique_id", "ds", f"y_lag_{lag}"]],
                on=["unique_id", "ds"],
                how="left",
            )
        # Check the values
        for lag in [1, 2]:
            pd.testing.assert_series_equal(
                result_df[f"y_lag_{lag}"], df_expected[f"y_lag_{lag}"]
            )

    def test_basic_lag_generation(self, sample_data, random_forest_model):
        """Test basic lag generation with gaps."""
        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1, 2])
        result_df = forecaster._generate_lags(
            sample_data.copy(), ds_col="ds", unique_id_col="unique_id", target_col="y"
        )

        # Manually create expected lags for comparison
        df_expected = sample_data.copy()
        for lag in [1, 2]:
            temp = sample_data.rename(columns={"y": f"y_lag_{lag}"})
            temp["ds"] = temp["ds"] + pd.to_timedelta(lag, unit="D")
            df_expected = pd.merge(
                df_expected,
                temp[["unique_id", "ds", f"y_lag_{lag}"]],
                on=["unique_id", "ds"],
                how="left",
            )

        # The _generate_lags function sorts the data, so we must too for comparison.
        result_df = result_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        df_expected = df_expected.sort_values(["unique_id", "ds"]).reset_index(
            drop=True
        )

        # Check that the lag columns are in the result
        assert "y_lag_1" in result_df.columns
        assert "y_lag_2" in result_df.columns

        # Check the values
        for col in df_expected.columns:
            pd.testing.assert_series_equal(result_df[col], df_expected[col])

    def test_lag_with_nan_in_target(self, sample_data, random_forest_model):
        """Test that lagging a NaN value results in a NaN."""
        data = sample_data.copy()
        # Introduce a NaN
        nan_index = 1
        series_id_with_nan = data.loc[nan_index, "unique_id"]
        date_with_nan = data.loc[nan_index, "ds"]
        data.loc[nan_index, "y"] = np.nan

        forecaster = GlobalForecaster(model=random_forest_model, freq="D", lags=[1])
        result_df = forecaster._generate_lags(
            data, ds_col="ds", unique_id_col="unique_id", target_col="y"
        )

        # Find the row for the next day and check if its lag is NaN
        next_day_lag = result_df[
            (result_df["unique_id"] == series_id_with_nan)
            & (result_df["ds"] == date_with_nan + pd.Timedelta(days=1))
        ]["y_lag_1"]

        if not next_day_lag.empty:
            assert pd.isna(next_day_lag.iloc[0])

    def test_no_lags_provided(self, sample_data, random_forest_model):
        """Test that the dataframe is returned unchanged if no lags are specified."""
        result_df = forecaster._generate_lags(
            sample_data.copy(), ds_col="ds", unique_id_col="unique_id", target_col="y"
        )
        assert "y_lag_1" not in result_df.columns

        expected_df = sample_data.sort_values(["unique_id", "ds"])
        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    def test_lag_longer_than_series(self, sample_data, random_forest_model):
        """Test that a lag longer than the series length results in all NaNs."""
        # Use a lag that is guaranteed to be longer than any series
        long_lag = 1000
        forecaster = GlobalForecaster(
            model=random_forest_model, freq="D", lags=[long_lag]
        )
        result_df = forecaster._generate_lags(
            sample_data.copy(), ds_col="ds", unique_id_col="unique_id", target_col="y"
        )
        assert result_df[f"y_lag_{long_lag}"].isna().all()

    def test_no_frequency_raises_error(self, sample_data, random_forest_model):
        """Test that a ValueError is raised if frequency is not set."""
        forecaster = GlobalForecaster(model=random_forest_model, lags=[1])
        forecaster._freq = None

        with pytest.raises(ValueError, match="Frequency must be set to generate lags"):
            forecaster._generate_lags(
                sample_data.copy(),
                ds_col="ds",
                unique_id_col="unique_id",
                target_col="y",
            )
