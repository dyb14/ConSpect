from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tables
from sklearn.preprocessing import minmax_scale

current_dir = Path(__file__).resolve().parent


@dataclass
class LoadData:
    data_files: list[str]
    holiday_file: str
    meteorology_file: str

    def __post_init__(self):
        self.DATA_PATHS = []
        for data_file in self.data_files:
            data_path = Path(data_file)
            if data_path.exists():
                self.DATA_PATHS.append(data_path)
            else:
                raise FileNotFoundError

        holiday_path = Path(self.holiday_file)
        if holiday_path.exists():
            self.HOLIDAY = holiday_path
        else:
            self.HOLIDAY = None

        meteorology_path = Path(self.meteorology_file)
        if meteorology_path.exists():
            self.METEOROLOGY = meteorology_path
        else:
            self.METEOROLOGY = None

    def load_holiday(self, timeslots: np.ndarray) -> Optional[np.ndarray]:
        if not self.HOLIDAY:
            return None

        timeslots = np.frompyfunc(lambda x: x[:8], 1, 1)(timeslots)

        with open(self.HOLIDAY, "r") as f:
            holidays = set([l.strip() for l in f])

        holidays = np.array(holidays)
        indices = np.where(np.isin(timeslots, holidays))[0]

        hv = np.zeros(len(timeslots))
        hv[indices] = 1.0
        return hv[:, np.newaxis]

    def load_meteorol(self, timeslots: np.ndarray):
        dat = tables.open_file(self.METEOROLOGY, mode="r")
        # dateformat: YYYYMMDD[slice]
        m_timeslots = dat.root.date.read().astype(str)
        wind_speed = dat.root.WindSpeed.read()
        weather = dat.root.Weather.read()
        temperature = dat.root.Temperature.read()
        dat.close()

        # Build a one-to-one mapping in the requested sample order. np.isin
        # selects rows in meteorology-file order and therefore returns extra
        # rows when that file contains duplicate timestamps (TaxiBJ contains
        # duplicated 2015-12-19 slots). The first occurrence is the valid
        # chronological occurrence; later duplicates form a repeated block.
        first_index_by_timestamp = {}
        for index, timestamp in enumerate(m_timeslots):
            first_index_by_timestamp.setdefault(timestamp, index)

        requested_timeslots = np.asarray(timeslots).astype(str)
        missing_timeslots = [
            timestamp
            for timestamp in requested_timeslots
            if timestamp not in first_index_by_timestamp
        ]
        if missing_timeslots:
            preview = ", ".join(missing_timeslots[:5])
            raise ValueError(
                f"meteorology data is missing {len(missing_timeslots)} requested "
                f"timeslot(s), first values: {preview}"
            )

        predicted_ids = np.fromiter(
            (first_index_by_timestamp[timestamp] for timestamp in requested_timeslots),
            dtype=np.int64,
            count=len(requested_timeslots),
        )
        if np.any(predicted_ids == 0):
            raise ValueError("meteorology data has no previous slot for a requested timestamp")
        cur_ids = predicted_ids - 1

        ws = wind_speed[cur_ids]
        wr = weather[cur_ids]
        te = temperature[cur_ids]

        # 0-1 scale
        ws = minmax_scale(ws)[:, np.newaxis]
        te = minmax_scale(te)[:, np.newaxis]

        # concatenate all these attributes
        merge_data = np.hstack([wr, ws, te])

        return merge_data

    def _remove_incomplete_days(
        self, dat: tables.file.File, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        # 20140425 has 24 timestamps, which does not appear in `incomplete_days` in the original implementation.
        # So I reimplemented it in a different way.
        data = dat.root.data.read()
        timestamps = dat.root.date.read().astype(str)

        dates, values = np.vstack(
            np.frompyfunc(lambda x: (x[:8], x[8:]), 1, 2)(timestamps)
        )
        # label encoding
        uniq_dates, labels = np.unique(dates, return_inverse=True)
        # groupby("labels")["values"].sum() != sum(range(1, 49))
        incomplete_days = uniq_dates[
            np.where(np.bincount(labels, values.astype(int)) != sum(range(1, (T + 1))))[
                0
            ]
        ]
        del_idx = np.where(np.isin(dates, incomplete_days))[0]
        new_data = np.delete(data, del_idx, axis=0)
        new_timestamps = np.delete(timestamps, del_idx)
        return new_data, new_timestamps

    def load_data(self, T: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        data_all = []
        timestamp_all = []
        for data_path in self.DATA_PATHS:
            dat = tables.open_file(data_path, mode="r")
            data, timestamps = self._remove_incomplete_days(dat, T=T)
            data[data < 0] = 0.0
            data_all.append(data)
            timestamp_all.append(timestamps)
            dat.close()

        return data_all, timestamp_all
