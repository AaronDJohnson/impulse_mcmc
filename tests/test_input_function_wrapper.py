import pytest
import numpy as np
from impulse.input_function_wrapper import _function_wrapper


class TestFunctionWrapper:
    """Test cases for _function_wrapper class"""
    
    def test_init_basic(self):
        """Test basic initialization"""
        def dummy_func(x):
            return np.sum(x)
        
        wrapper = _function_wrapper(dummy_func)
        assert wrapper.f is dummy_func
        assert wrapper.args == ()
        assert wrapper.kwargs == {}
        assert wrapper.vectorized is False
    
    def test_init_with_args_kwargs(self):
        """Test initialization with args and kwargs"""
        def dummy_func(x, a, b=1):
            return np.sum(x) + a + b
        
        args = (5,)
        kwargs = {'b': 10}
        wrapper = _function_wrapper(dummy_func, args, kwargs, vectorized=True)
        
        assert wrapper.args == args
        assert wrapper.kwargs == kwargs
        assert wrapper.vectorized is True
    
    def test_call_2d_input_required(self):
        """Test that input must be 2D"""
        def dummy_func(x):
            return np.sum(x)
        
        wrapper = _function_wrapper(dummy_func)
        
        # 1D input should raise error
        with pytest.raises(ValueError, match="Input to _function_wrapper must be 2-D"):
            wrapper(np.array([1, 2, 3]))
        
        # 3D input should raise error
        with pytest.raises(ValueError, match="Input to _function_wrapper must be 2-D"):
            wrapper(np.array([[[1, 2], [3, 4]]]))
    
    def test_call_nonzero_leading_dim(self):
        """Test that leading dimension must be nonzero"""
        def dummy_func(x):
            return np.sum(x)
        
        wrapper = _function_wrapper(dummy_func)
        
        # Empty array should raise error
        with pytest.raises(ValueError, match="Input to _function_wrapper must have nonzero leading dimension"):
            wrapper(np.zeros((0, 2)))
    
    def test_call_non_vectorized(self):
        """Test non-vectorized function calls"""
        def simple_func(x):
            return np.sum(x**2)
        
        wrapper = _function_wrapper(simple_func)
        
        input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapper(input_data)
        
        expected = np.array([5.0, 25.0, 61.0])  # 1²+2², 3²+4², 5²+6²
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_call_vectorized(self):
        """Test vectorized function calls"""
        def vectorized_func(x):
            return np.sum(x**2, axis=1)
        
        wrapper = _function_wrapper(vectorized_func, vectorized=True)
        
        input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapper(input_data)
        
        expected = np.array([5.0, 25.0, 61.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_call_with_args_kwargs(self):
        """Test function calls with additional args/kwargs"""
        def func_with_params(x, scale=1.0, offset=0.0):
            return scale * np.sum(x) + offset
        
        wrapper = _function_wrapper(func_with_params, kwargs={'scale': 2.0, 'offset': 1.0})
        
        input_data = np.array([[1.0, 1.0]])
        result = wrapper(input_data)
        
        expected = np.array([5.0])  # 2.0 * (1+1) + 1.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_vectorized_wrong_output_shape(self):
        """Test vectorized function with wrong output shape"""
        def bad_vectorized_func(x):
            # Returns wrong shape - should be (n,) but returns (1,)
            return np.array([np.sum(x)])
        
        wrapper = _function_wrapper(bad_vectorized_func, vectorized=True)
        
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="Vectorized function returned array with incorrect leading dimension"):
            wrapper(input_data)
    
    def test_repr(self):
        """Test string representation"""
        def named_func(x):
            return x
        
        wrapper = _function_wrapper(named_func, vectorized=True)
        repr_str = repr(wrapper)
        
        assert "named_func" in repr_str
        assert "vectorized=True" in repr_str
    
    def test_extra_args_application(self):
        """Test that extra args/kwargs are passed correctly (x first, then args)"""
        def func_with_multiple_args(x, a, b, c=1):
            return np.sum(x) + a + b + c

        wrapper = _function_wrapper(
            func_with_multiple_args,
            args=(10, 20),
            kwargs={'c': 5}
        )

        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = wrapper(input_data)

        # Call is: func(x_row, 10, 20, c=5) for each row
        # Row [1.0, 2.0]: sum([1.0, 2.0]) + 10 + 20 + 5 = 3 + 35 = 38
        # Row [3.0, 4.0]: sum([3.0, 4.0]) + 10 + 20 + 5 = 7 + 35 = 42
        expected = np.array([38.0, 42.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_single_row_input(self):
        """Test with single row input"""
        def simple_func(x):
            return np.sum(x)
        
        wrapper = _function_wrapper(simple_func)
        
        input_data = np.array([[1.0, 2.0, 3.0]])
        result = wrapper(input_data)
        
        expected = np.array([6.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_threaded_matches_sequential(self):
        """Threaded and sequential paths produce identical results"""
        def simple_func(x):
            return np.sum(x**2)

        input_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        wrapper_seq = _function_wrapper(simple_func, threads=1)
        wrapper_thr = _function_wrapper(simple_func, threads=2)

        result_seq = wrapper_seq(input_data)
        result_thr = wrapper_thr(input_data)

        np.testing.assert_array_almost_equal(result_seq, result_thr)

    def test_threaded_skipped_when_vectorized(self):
        """Executor not created when vectorized=True"""
        def vectorized_func(x):
            return np.sum(x**2, axis=1)

        wrapper = _function_wrapper(vectorized_func, vectorized=True, threads=4)
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        wrapper(input_data)

        assert wrapper._executor is None

    def test_threaded_skipped_for_single_row(self):
        """No threading for n=1 input"""
        def simple_func(x):
            return np.sum(x**2)

        wrapper = _function_wrapper(simple_func, threads=4)
        input_data = np.array([[1.0, 2.0]])
        wrapper(input_data)

        assert wrapper._executor is None

    def test_getstate_drops_executor(self):
        """__getstate__ drops executor, wrapper works after restore"""
        def simple_func(x):
            return np.sum(x**2)

        wrapper = _function_wrapper(simple_func, threads=2)
        input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Force executor creation
        wrapper(input_data)
        assert wrapper._executor is not None

        # __getstate__ should drop executor
        state = wrapper.__getstate__()
        assert state['_executor'] is None
        assert state['threads'] == 2

        # Simulate unpickle by creating new wrapper and restoring state
        wrapper2 = object.__new__(_function_wrapper)
        wrapper2.__dict__.update(state)

        assert wrapper2._executor is None
        assert wrapper2.threads == 2

        # Still works after restore (executor lazily recreated)
        result = wrapper2(input_data)
        expected = np.array([5.0, 25.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_complex_function(self):
        """Test with more complex mathematical function"""
        def complex_func(x):
            # Rosenbrock function in 2D
            return -(100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)
        
        wrapper = _function_wrapper(complex_func)
        
        input_data = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 4.0]])
        result = wrapper(input_data)
        
        expected = np.array([0.0, -1.0, -1.0])  # Rosenbrock values: [1,1]=0, [0,0]=-1, [2,4]=-1  
        np.testing.assert_array_almost_equal(result, expected)