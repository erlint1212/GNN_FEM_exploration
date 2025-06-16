import unittest
import os
import shutil
import numpy as np
import torch
import networkx as nx
import matplotlib # Prevents plt.show() from blocking in non-interactive environments for tests
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt


# Assuming plot_funcs.py is in GAT_r_adaptivity folder
from GAT_r_adaptivity import plot_funcs

class TestPlotFuncs(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_output_dir = "test_plot_outputs"
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir) # Clean up from previous runs
        os.makedirs(cls.test_output_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)

    def _get_output_path(self, func_name):
        # Construct a unique filename within the test_output_dir for each function
        # Note: The actual plot_funcs save with datetime, so we just check if *any* file is made.
        # This is a placeholder for a more robust check if needed.
        return self.test_output_dir

    def test_density_plot_matrix(self):
        path = self._get_output_path("density_plot_matrix")
        
        # Test with 2D numpy array
        matrix_2d = np.random.rand(10, 2)
        plot_funcs.density_plot_matrix(matrix_2d, output=path, title="TestDensity2D", show=False)
        self.assertTrue(any(fname.startswith("TestDensity2D_") for fname in os.listdir(path)))
        
        # Test with 1D numpy array
        matrix_1d = np.random.rand(10)
        plot_funcs.density_plot_matrix(matrix_1d, output=path, title="TestDensity1D", show=False)
        self.assertTrue(any(fname.startswith("TestDensity1D_") for fname in os.listdir(path)))

        # Test with empty array
        plot_funcs.density_plot_matrix(np.array([]), output=path, title="TestDensityEmpty", show=False)
        # Should not create a file, but also not error. Check no new file with this title.
        
        # Test with non-numpy list
        list_data = [[1,2],[3,4]]
        plot_funcs.density_plot_matrix(list_data, output=path, title="TestDensityList", show=False)
        self.assertTrue(any(fname.startswith("TestDensityList_") for fname in os.listdir(path)))
        
        # Clean up specific files if needed or rely on tearDownClass for the directory
        for f in os.listdir(path): # Basic cleanup of specific files if created
            if "TestDensity" in f: os.remove(os.path.join(path, f))


    def test_loss_plot(self):
        path = self._get_output_path("loss_plot")
        epochs = list(range(1, 11))
        train_loss = [1/x for x in epochs]
        test_loss = [1.2/x for x in epochs]
        plot_funcs.loss_plot(epochs, train_loss, test_loss, model_name="TestModelLoss", output=path, show=False)
        # Check if a file matching the pattern is created
        self.assertTrue(any(fname.startswith("TestModelLoss_loss_plot_") for fname in os.listdir(path)))
        for f in os.listdir(path):
            if "TestModelLoss_loss_plot_" in f: os.remove(os.path.join(path, f))


    def test_pred_vs_true(self):
        path = self._get_output_path("predVStrue")
        true_val = [np.array([1,2,3]), np.array([4,5,6])]
        pred_val = [np.array([1.1,2.1,3.1]), np.array([3.9,5.0,6.1])]
        true_train = [np.array([0.5,1.5]), np.array([2.5])]
        pred_train = [np.array([0.6,1.4]), np.array([2.6])]

        plot_funcs.predVStrue(true_val, pred_val, true_train, pred_train, model_name="TestPredTrue", output=path, show=False)
        self.assertTrue(any(fname.startswith("TestPredTrue_predVStrue_") for fname in os.listdir(path)))
        for f in os.listdir(path):
            if "TestPredTrue_predVStrue_" in f: os.remove(os.path.join(path, f))

        # Test with empty training data
        plot_funcs.predVStrue(true_val, pred_val, [], [], model_name="TestPredTrueNoTrain", output=path, show=False)
        self.assertTrue(any(fname.startswith("TestPredTrueNoTrain_predVStrue_") for fname in os.listdir(path)))
        for f in os.listdir(path):
             if "TestPredTrueNoTrain_predVStrue_" in f: os.remove(os.path.join(path, f))


    def test_cuda_status(self):
        # This function mainly prints. We'll just check it runs without error.
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            plot_funcs.cuda_status(device)
        except Exception as e:
            self.fail(f"cuda_status raised an exception: {e}")

    def test_plot_accuracy_vs_cost(self):
        # This function currently calls plt.show() directly, which is problematic for non-interactive tests.
        # For robust testing, it should ideally also accept output path and show=False.
        # We'll mock plt.show() and plt.savefig() to verify calls for now.
        # Assuming future modification or testing its execution without error.
        classical_costs = [100, 200, 300]
        classical_accuracies = [0.1, 0.05, 0.02]
        gat_costs = [110, 210, 310]
        gat_accuracies = [0.09, 0.04, 0.01]
        try:
            with unittest.mock.patch('matplotlib.pyplot.show') as mock_show:
                 plot_funcs.plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies)
                 mock_show.assert_called_once() # if show is True by default
        except Exception as e:
            self.fail(f"plot_accuracy_vs_cost raised an exception: {e}")


    def test_plot_time_comparison(self):
        path = self._get_output_path("plot_time_comparison")
        classical_times = [10.5, 12.3, 11.0]
        gat_times = [0.5, 0.6, 0.55]
        
        # Test bar plot
        plot_funcs.plot_time_comparison(classical_times, gat_times, use_box_plot=False, output=path, title="TimeBar", show=False)
        self.assertTrue(any(fname.startswith("TimeBar_time_comp_") for fname in os.listdir(path)))
        for f in os.listdir(path):
            if "TimeBar_time_comp_" in f: os.remove(os.path.join(path,f))

        # Test box plot
        plot_funcs.plot_time_comparison(classical_times, gat_times, use_box_plot=True, output=path, title="TimeBox", show=False)
        self.assertTrue(any(fname.startswith("TimeBox_time_comp_") for fname in os.listdir(path)))
        for f in os.listdir(path):
            if "TimeBox_time_comp_" in f: os.remove(os.path.join(path,f))
            
    def test_plot_convergence(self):
        # Similar to plot_accuracy_vs_cost, test execution and mock plt.show
        classical_dofs = [100, 1000, 10000]
        classical_errors = [0.1, 0.01, 0.001]
        gat_dofs = [120, 1100, 10500]
        gat_errors = [0.09, 0.009, 0.0009]
        try:
            with unittest.mock.patch('matplotlib.pyplot.show') as mock_show:
                plot_funcs.plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors)
                mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_convergence raised an exception: {e}")

    def test_plot_gat_training_loss(self):
        # Similar to plot_accuracy_vs_cost, test execution and mock plt.show
        epochs = list(range(1,11))
        training_loss = [1/e for e in epochs]
        validation_loss = [1.2/e for e in epochs]
        try:
            with unittest.mock.patch('matplotlib.pyplot.show') as mock_show:
                plot_funcs.plot_gat_training_loss(epochs, training_loss, validation_loss=validation_loss)
                mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_gat_training_loss raised an exception: {e}")

    def test_weisfeiler_lehman_test(self):
        G = nx.Graph()
        G.add_edges_from([(0,1), (1,2), (2,0), (2,3), (3,4)])
        try:
            history = plot_funcs.weisfeiler_lehman_test(G, iterations=2)
            self.assertIsInstance(history, list)
            self.assertEqual(len(history), 2 + 1) # iterations + initial
            for item in history:
                self.assertIsInstance(item, plot_funcs.Counter)
        except Exception as e:
            self.fail(f"weisfeiler_lehman_test raised an exception: {e}")

    def test_plot_wl_histograms(self):
        G = nx.Graph()
        G.add_edges_from([(0,1), (1,2)])
        history = plot_funcs.weisfeiler_lehman_test(G, iterations=1)
        try:
            with unittest.mock.patch('matplotlib.pyplot.show') as mock_show:
                plot_funcs.plot_wl_histograms(history, graph_name="TestWLGraph")
                mock_show.assert_called_once()
        except Exception as e:
            self.fail(f"plot_wl_histograms raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
