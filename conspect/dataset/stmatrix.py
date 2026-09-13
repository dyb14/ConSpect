import numpy as np
from tqdm import tqdm

# The original implementation uses a lot of loops, which slows down the execution time.
# Therefore, I reimplemented to reduce loops as much as possible.


class STMatrix:
    def __init__(self, data: np.ndarray, timestamps: np.ndarray, T: int) -> None:
        if len(data) != len(timestamps):
            raise ValueError("data and timestamps must have the same length")
        if T <= 0:
            raise ValueError("T must be positive")
        self.data = data
        self.timestamps = timestamps
        self.T = int(T)
        self.np_timestamps = self._string2timestamp(timestamps)
        self.slot_delta = np.timedelta64(round(86400 / self.T), "s")
        self._index_by_timestamp = {
            timestamp.astype("datetime64[s]").astype(np.int64).item(): index
            for index, timestamp in enumerate(self.np_timestamps)
        }

    def _string2timestamp(self, timestamps: np.ndarray) -> np.ndarray:
        parsed = []
        for raw_timestamp in np.asarray(timestamps).astype(str):
            if len(raw_timestamp) < 9:
                raise ValueError(f"invalid timeslot timestamp: {raw_timestamp!r}")
            slot = int(raw_timestamp[8:])
            if not 1 <= slot <= self.T:
                raise ValueError(
                    f"timeslot {slot} is outside [1, {self.T}] for "
                    f"timestamp {raw_timestamp!r}"
                )
            midnight = np.datetime64(
                f"{raw_timestamp[:4]}-{raw_timestamp[4:6]}-{raw_timestamp[6:8]}",
                "s",
            )
            seconds_after_midnight = round((slot - 1) * 86400 / self.T)
            parsed.append(midnight + np.timedelta64(seconds_after_midnight, "s"))
        return np.asarray(parsed, dtype="datetime64[s]")

    def get_indices(self, timestamps: np.ndarray) -> np.ndarray:
        indices = []
        for timestamp in np.asarray(timestamps, dtype="datetime64[s]"):
            key = timestamp.astype(np.int64).item()
            if key not in self._index_by_timestamp:
                raise KeyError(f"timestamp {timestamp} is not present in the data")
            indices.append(self._index_by_timestamp[key])
        return np.asarray(indices, dtype=np.int64)

    def get_matrices(self, timestamps: np.ndarray) -> np.ndarray:
        indices = self.get_indices(timestamps)
        return self.data[indices, :, :, :]

    def create_cpt_matrices(
        self,
        len_closeness: int = 3,
        len_period: int = 1,
        len_trend: int = 1,
        map_height: int = 32,
        map_width: int = 32,
    ) -> dict[str, np.ndarray]:
        """Create C/P/T samples by exact timestamp lookup.

        Closeness uses the immediately preceding slots at this dataset's time
        resolution. Period and trend use the same wall-clock time on preceding
        days and weeks. A target is kept only when every requested timestamp is
        present, so missing slots never silently shift a context window.
        """
        lengths = (len_closeness, len_period, len_trend)
        if any(length <= 0 for length in lengths):
            raise ValueError("C/P/T lengths must all be positive")
        expected_spatial_shape = (map_height, map_width)
        if tuple(self.data.shape[-2:]) != expected_spatial_shape:
            raise ValueError(
                f"data spatial shape {tuple(self.data.shape[-2:])} does not "
                f"match {expected_spatial_shape}"
            )

        x_c, x_p, x_t, y, ts_y = [], [], [], [], []
        day = np.timedelta64(1, "D")
        week = np.timedelta64(7, "D")
        for target_index, target_time in enumerate(
            tqdm(self.np_timestamps, dynamic_ncols=True)
        ):
            closeness_times = np.asarray(
                [
                    target_time - lag * self.slot_delta
                    for lag in range(len_closeness, 0, -1)
                ]
            )
            period_times = np.asarray(
                [target_time - lag * day for lag in range(len_period, 0, -1)]
            )
            trend_times = np.asarray(
                [target_time - lag * week for lag in range(len_trend, 0, -1)]
            )
            context_times = (closeness_times, period_times, trend_times)
            if not all(
                all(
                    timestamp.astype("datetime64[s]").astype(np.int64).item()
                    in self._index_by_timestamp
                    for timestamp in requested_times
                )
                for requested_times in context_times
            ):
                continue

            closeness = self.get_matrices(closeness_times)
            period = self.get_matrices(period_times)
            trend = self.get_matrices(trend_times)
            x_c.append(closeness.reshape(-1, map_height, map_width))
            x_p.append(period.reshape(-1, map_height, map_width))
            x_t.append(trend.reshape(-1, map_height, map_width))
            y.append(self.data[target_index])
            ts_y.append(self.timestamps[target_index])

        return {
            "closeness": np.asarray(x_c),
            "period": np.asarray(x_p),
            "trend": np.asarray(x_t),
            "y": np.asarray(y),
            "timestamps_y": np.asarray(ts_y),
        }
