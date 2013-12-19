import unittest
from mozsci import variables


class TestVariable(unittest.TestCase):
    """
    Test cases for dumps_variable_def.
    """
    def test_to_str_1(self):
        """
        Test the __str__ method of the Variable class.
        """
        variable = variables.Variable(name='test1', normalization=True, mean_std=(10, 0.5))
        var_str = str(variable)
        expected = '''name: test1, normalization: True, mean_std: (10, 0.5), description: None'''

        self.assertEqual(var_str, expected)

    def test_dumps_json_1(self):
        """
        Test the json string for parameters dumped from the variable definition.
        """
        variable = variables.Variable(name='test1', normalization=True, mean_std=(10, 0.5))
        dumped = variable.dump_parameters()
        expected = '''{"mean_std": [10, 0.5], "normalization": true}'''

        self.assertEqual(dumped, expected)

    def test_loads_json_1(self):
        """
        Test the json string for parameters dumped from the variable definition.
        """
        variable = variables.Variable(name='test1', description='for testing')
        json_str = '''{"mean_std": [10, 0.5], "normalization": true}'''

        variable.load_parameters(json_str)

        variable_str = str(variable)

        expected = 'name: test1, normalization: True, mean_std: [10, 0.5], description: for testing'

        self.assertEqual(variable_str, expected)


class TestModelVariables(unittest.TestCase):
    """
    Test cases for dumps_variable_def.
    """
    def test_to_str_1(self):
        """
        Test the __str__ method of the Variable class.
        """
        var1 = variables.Variable(name='test1', normalization=True, mean_std=(10, 0.5))
        var2 = variables.Variable(name='test2', normalization=False, description='test case use')
        var3 = variables.Variable(name='test3', normalization=False)

        model_vars = variables.ModelVariables(independent=[var2, var3], dependent=[var1], schema=[var1, var2, var3])
        model_var_str = str(model_vars)
        expected = '''indepdent: [name: test2, normalization: False, mean_std: None, description: test case use; name: test3, normalization: False, mean_std: None, description: None], dependent: [name: test1, normalization: True, mean_std: (10, 0.5), description: None], schema [name: test1, normalization: True, mean_std: (10, 0.5), description: None; name: test2, normalization: False, mean_std: None, description: test case use; name: test3, normalization: False, mean_std: None, description: None]'''

        self.assertEqual(model_var_str, expected)

    def test_data_str_1(self):
        """
        Test the data_str method of the Variable class.
        """
        var1 = variables.Variable(name='test1', pre_transform=lambda x: str(x[1]))
        var2 = variables.Variable(name='test2', pre_transform=lambda x: str(x[2]))
        var3 = variables.Variable(name='test3', pre_transform=lambda x: str(x[0]))

        model_vars = variables.ModelVariables(independent=[var2, var3], dependent=[var1], schema=[var1, var2, var3])
        output = model_vars.data_str([100, 200, 300])
        expected = '200	300	100'

        self.assertEqual(output, expected)

    def test_dump_parameters_1(self):
        """
        Test the dump parameter method of the Variable class.
        """
        var1 = variables.Variable(name='test1', pre_transform=lambda x: str(x[1]), normalization=True, mean_std=(1, 2))
        var2 = variables.Variable(name='test2', pre_transform=lambda x: str(x[2]), normalization=False)
        var3 = variables.Variable(name='test3', pre_transform=lambda x: str(x[0]), normalization=True, mean_std=(2.0, 0.3))

        model_vars = variables.ModelVariables(independent=[var2, var3], dependent=[var1], schema=[var1, var2, var3])
        output = model_vars.dump_parameters()
        expected = '{"dependent": "[\\"{\\\\\\"mean_std\\\\\\": [1, 2], \\\\\\"normalization\\\\\\": true}\\"]", "independent": "[\\"{\\\\\\"mean_std\\\\\\": null, \\\\\\"normalization\\\\\\": false}\\", \\"{\\\\\\"mean_std\\\\\\": [2.0, 0.3], \\\\\\"normalization\\\\\\": true}\\"]", "schema": "[\\"{\\\\\\"mean_std\\\\\\": [1, 2], \\\\\\"normalization\\\\\\": true}\\", \\"{\\\\\\"mean_std\\\\\\": null, \\\\\\"normalization\\\\\\": false}\\", \\"{\\\\\\"mean_std\\\\\\": [2.0, 0.3], \\\\\\"normalization\\\\\\": true}\\"]"}'

        self.assertEqual(output, expected)

    def test_load_parameters_1(self):
        """
        Test the dump parameter method of the Variable class.
        """
        var1 = variables.Variable(name='test1', pre_transform=lambda x: str(x[1]), normalization=True, mean_std=(2.0, 3.0))
        var2 = variables.Variable(name='test2', pre_transform=lambda x: str(x[2]), normalization=False)
        var3 = variables.Variable(name='test3', pre_transform=lambda x: str(x[0]), normalization=False)

        model_vars = variables.ModelVariables(independent=[var2, var3], dependent=[var1], schema=[var1, var2, var3])
        j_str = '{"dependent": "[\\"{\\\\\\"mean_std\\\\\\": [1, 2], \\\\\\"normalization\\\\\\": true}\\"]", "independent": "[\\"{\\\\\\"mean_std\\\\\\": null, \\\\\\"normalization\\\\\\": false}\\", \\"{\\\\\\"mean_std\\\\\\": [2.0, 0.3], \\\\\\"normalization\\\\\\": true}\\"]", "schema": "[\\"{\\\\\\"mean_std\\\\\\": [1, 2], \\\\\\"normalization\\\\\\": true}\\", \\"{\\\\\\"mean_std\\\\\\": null, \\\\\\"normalization\\\\\\": false}\\", \\"{\\\\\\"mean_std\\\\\\": [2.0, 0.3], \\\\\\"normalization\\\\\\": true}\\"]"}'
        model_vars.load_parameters(j_str)

        # now var1 and var3 should have the new normalization settings.
        self.assertEqual(var1.mean_std, [1, 2])
        self.assertTrue(var3.normalization)
        self.assertEqual(var3.mean_std, [2.0, 0.3])


if __name__ == "__main__":
    unittest.main()
