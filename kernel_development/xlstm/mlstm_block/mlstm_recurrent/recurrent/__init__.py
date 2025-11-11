#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from .native_step import mlstm_recurrent_step__native

try:
    from .native_sequence import (
        mlstm_recurrent_sequence__native_fw,
        mlstm_recurrent_sequence__triton_alternate_step_fw,
        mlstm_recurrent_sequence__triton_step_fused_fw,
    )
    from .triton_step import mlstm_recurrent_step__triton
    from .triton_step_alternate import mlstm_recurrent_step__triton_alternate

    # Use compiled PyTorch graph implementation for metal step
    from .metal.compiled import mlstm_recurrent_step__metal_fw, mlstm_recurrent_step__metal

    registry_step = {
        "native": mlstm_recurrent_step__native,
        # "triton_alternate": mlstm_recurrent_step__triton_alternate,
        "triton": mlstm_recurrent_step__triton,
        "metal": mlstm_recurrent_step__metal,
    }

    from .native_sequence import mlstm_recurrent_sequence__native_fw as _native_seq
    from .native_sequence import _mlstm_recurrent_sequence_loop_fw as _loop


    def mlstm_recurrent_sequence__metal_fw(
            q, k, v, i, f,
            c_initial=None, n_initial=None, m_initial=None,
            return_last_states: bool = False,
            eps: float = 1e-6,
            dtype_state: 'torch.dtype' = None,
            **kwargs,
    ):
        """

        :param q:
        :param k:
        :param v:
        :param i:
        :param f:
        :param c_initial:
        :param n_initial:
        :param m_initial:
        :param return_last_states:
        :param eps:
        :param dtype_state:
        :param kwargs:
        :return:
        """
        import torch as _t
        if dtype_state is None:
            dtype_state = _t.float32
        ret = _loop(mlstm_step_fn=mlstm_recurrent_step__metal_fw, matQ=q, matK=k, matV=v, vecI=i, vecF=f,
                    matC_initial=c_initial, vecN_initial=n_initial, scaM_initial=m_initial,
                    return_last_states=return_last_states, eps=eps, dtype_state=dtype_state)
        # ret = (matH, last_states or None, all_states or None)
        if return_last_states:
            return ret[0], ret[1]
        else:
            return ret[0]


    registry_sequence = {
        "native_sequence__native": mlstm_recurrent_sequence__native_fw,
        # "native_sequence__triton_alternate": mlstm_recurrent_sequence__triton_alternate_step_fw,
        "native_sequence__triton": mlstm_recurrent_sequence__triton_step_fused_fw,
        "native_sequence__metal": mlstm_recurrent_sequence__metal_fw,
    }
except ImportError:
    from .native_sequence import mlstm_recurrent_sequence__native_fw

    from .metal.compiled import mlstm_recurrent_step__metal_fw, mlstm_recurrent_step__metal

    registry_step = {
        "native": mlstm_recurrent_step__native,
        "metal": mlstm_recurrent_step__metal,
    }

    from .native_sequence import _mlstm_recurrent_sequence_loop_fw as _loop


    def mlstm_recurrent_sequence__metal_fw(
            q, k, v, i, f,
            c_initial=None, n_initial=None, m_initial=None,
            return_last_states: bool = False,
            eps: float = 1e-6,
            dtype_state: 'torch.dtype' = None,
            **kwargs,
    ):
        """

        :param q:
        :param k:
        :param v:
        :param i:
        :param f:
        :param c_initial:
        :param n_initial:
        :param m_initial:
        :param return_last_states:
        :param eps:
        :param dtype_state:
        :param kwargs:
        :return:
        """
        import torch as _t
        if dtype_state is None:
            dtype_state = _t.float32
        ret = _loop(mlstm_step_fn=mlstm_recurrent_step__metal_fw, matQ=q, matK=k, matV=v, vecI=i, vecF=f,
                    matC_initial=c_initial, vecN_initial=n_initial, scaM_initial=m_initial,
                    return_last_states=return_last_states, eps=eps, dtype_state=dtype_state)
        if return_last_states:
            return ret[0], ret[1]
        else:
            return ret[0]


    registry_sequence = {
        "native_sequence__native": mlstm_recurrent_sequence__native_fw,
        "native_sequence__metal": mlstm_recurrent_sequence__metal_fw,
    }
