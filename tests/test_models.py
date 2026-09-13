import unittest

import torch

from conspect.model import ConSpect


class ConSpectForwardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 2
        self.height = 8
        self.width = 8
        self.nb_flow = 2
        self.external_dim = 5
        self.x_c = torch.randn(self.batch, 3 * self.nb_flow, self.height, self.width)
        self.x_p = torch.randn(self.batch, 1 * self.nb_flow, self.height, self.width)
        self.x_t = torch.randn(self.batch, 1 * self.nb_flow, self.height, self.width)
        self.ext = torch.randn(self.batch, self.external_dim)

    def _model(self):
        return ConSpect(
            len_closeness=3,
            len_period=1,
            len_trend=1,
            external_dim=self.external_dim,
            nb_flow=self.nb_flow,
            map_height=self.height,
            map_width=self.width,
            nb_residual_unit=2,
        ).eval()

    def test_forward_shape(self):
        model = self._model()
        with torch.no_grad():
            out = model(self.x_c, self.x_p, self.x_t, self.ext)
        self.assertEqual(tuple(out.shape), (self.batch, 2, self.height, self.width))
        self.assertTrue(torch.isfinite(out).all())

    def test_debug_outputs(self):
        model = self._model()
        with torch.no_grad():
            out, debug = model.forward_with_debug(self.x_c, self.x_p, self.x_t, self.ext)
        self.assertEqual(tuple(out.shape), (self.batch, 2, self.height, self.width))
        self.assertEqual(tuple(debug["state_scale"].shape), (self.batch,))
        self.assertEqual(tuple(debug["effective_tau"].shape), (self.batch, 2))
        self.assertEqual(tuple(debug["fusion_alpha"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
