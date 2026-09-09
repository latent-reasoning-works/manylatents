"""Snapshot routing with deterministic fake backend state, without any fits."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from numpy.testing import assert_allclose, assert_array_equal

from manylatents.metrics.operator_snapshot import OperatorSnapshot
from manylatents.algorithms.latent.umap import UMAPModule
from manylatents.algorithms.latent.tsne import TSNEModule
from manylatents.algorithms.latent.phate import PHATEModule

X = np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
Y = X[:,:2].copy()
K = np.array([[0.,.8,.2],[.8,0,.4],[.2,.4,0]],dtype=np.float32)


def native_stub(cls):
    module = object.__new__(cls)
    module._is_fitted=True; module._resolved_backend=None; module.fit_fraction=1.
    module.n_components=2; module.random_state=1701; module.init_seed=1701
    module.backend='sklearn'; module.device='cpu'; module.neighborhood_size=15
    return module


def umap_stub():
    module = native_stub(UMAPModule)
    for name,value in dict(n_neighbors=15,min_dist=.5,n_epochs=200,learning_rate=1.,negative_sample_rate=5,metric='euclidean').items():
        setattr(module,name,value)
    backend_type = type('UMAP',(),{'__module__':'umap.umap_','get_params':lambda self,deep=False:dict(n_neighbors=15,random_state=1701)})
    module.model=backend_type()
    module.model.graph_=csr_matrix(K)
    module.model.embedding_=Y.copy()
    return module


def tsne_stub():
    module = native_stub(TSNEModule)
    for name,value in dict(perplexity=15.,n_iter_early=250,n_iter_late=750,learning_rate='auto',initialization='random',metric='euclidean').items():
        setattr(module,name,value)
    backend_type=type('PerplexityBasedNN',(),{'__module__':'openTSNE.affinity'})
    module.affinities=backend_type()
    module.affinities.P=csr_matrix(K/K.sum())
    module.affinities.perplexity=15.; module.affinities.effective_perplexity_=15.
    module.affinities.symmetrize=True; module.affinities.n_jobs=-1
    module.affinities.knn_index=SimpleNamespace(k=2)
    class Reference(np.ndarray):
        def transform(self,X,perplexity=5,initialization='median',k=25,learning_rate=.1,n_iter=250):
            raise AssertionError('Snapshot must not transform')
    module.embedding_train=np.array(Y,dtype=np.float64).view(Reference)
    return module


def phate_stub():
    module=native_stub(PHATEModule)
    for name,value in dict(knn=15,t=5,gamma=1,decay=40,n_landmark=None,n_pca=None,n_jobs=4,verbose=False).items():
        setattr(module,name,value)
    backend_type=type('PHATE',(),{'__module__':'phate.phate','get_params':lambda self:dict(knn=15,t=5,n_landmark=None),
                                 'diff_potential':property(lambda self:self._diff_potential)})
    module.model=backend_type()
    module.model.graph=SimpleNamespace(K=csr_matrix(K+np.eye(3)),get_params=lambda:dict(anisotropy=0,kernel_symm='+'))
    module.model.diff_op=(K+np.eye(3))/(K+np.eye(3)).sum(axis=1,keepdims=True)
    module.model._diff_potential=-np.log(np.linalg.matrix_power(module.model.diff_op,5)+1e-7)
    module.model.n_landmark=None; module.model.t=5; module.model.gamma=1; module.model.optimal_t=None
    module.model.embedding=Y.copy()
    return module


@pytest.mark.parametrize('factory,method',[(umap_stub,'umap'),(tsne_stub,'tsne'),(phate_stub,'phate')])
def test_copies_objects_dtypes_and_stage_labels(factory,method):
    module=factory()
    result=OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    assert result['metadata']['method']==method
    assert result['input_data'].dtype==np.float32
    assert result['input_effective'].dtype==np.float64
    assert result['returned_embedding'].dtype==np.float32
    assert not np.shares_memory(result['returned_embedding'],Y)
    assert not np.shares_memory(result['input_data'],X)
    assert_allclose(result['affinity'].sum(axis=1),1,atol=1e-6)
    assert result['metadata']['native_graph_storage']['representation']=='csr'
    if method=='tsne':
        assert result['optimized_reference_after_transform'].dtype==np.float64
        assert result['metadata']['realized']['transform_defaults']['perplexity']==5
    if method=='phate':
        assert_allclose(np.exp(-result['phate_potential'])-1e-7,np.linalg.matrix_power(result['affinity'],5))
        assert np.all(np.diag(result['kernel'])==1)


def test_hydra_array_preservation_without_spectral_prewarm(monkeypatch):
    from manylatents.api import _resolve_metrics
    import importlib
    evaluator=importlib.import_module('manylatents.evaluate')
    def no_spectrum(*args,**kwargs):
        raise AssertionError('Snapshot must not trigger spectral prewarming')
    monkeypatch.setattr(evaluator,'compute_eigenvalues',no_spectrum)
    metrics,_=_resolve_metrics({'snapshot':{'_target_':'manylatents.metrics.operator_snapshot.OperatorSnapshot',
                                          '_partial_':True,'at':'embedding'}})
    result=evaluator.evaluate(Y,dataset=SimpleNamespace(data=X),module=umap_stub(),metrics=metrics)
    assert_array_equal(result['snapshot.kernel'],K)
    assert isinstance(result['snapshot.metadata'],dict)
    assert result['snapshot.returned_embedding'].shape==(3,2)


def test_failure_domains_and_no_missing_phate_potential():
    module=umap_stub(); module._resolved_backend='torchdr'
    with pytest.raises(ValueError,match='CPU'):
        OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    module=umap_stub(); module.fit_fraction=.5
    with pytest.raises(ValueError,match='fit_fraction'):
        OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    module=phate_stub(); module.model._diff_potential=None
    with pytest.raises(ValueError,match='potential'):
        OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    module=phate_stub(); module.n_landmark=2
    with pytest.raises(ValueError,match='Landmark'):
        OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    module=umap_stub(); module.model.graph_=csr_matrix(np.ones((2,2)))
    with pytest.raises(ValueError,match='n particle'):
        OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)


def test_explicit_sparse_zeros_survive_metadata_and_captures_are_independent():
    module=umap_stub()
    graph=csr_matrix((np.array([0.,.5,.5,.5,.5]),
                      np.array([0,1,0,2,1]),np.array([0,2,4,5])),shape=(3,3))
    module.model.graph_=graph
    result=OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    assert result['metadata']['native_graph_storage']['explicit_zero_entries']==1
    assert result['metadata']['native_graph_storage']['stored_entries']==5
    graph.data[:]=9
    assert result['kernel'][0,0]==0
    assert result['kernel'][0,1]==.5


def test_snapshot_zero_row_is_preserved_for_downstream_unavailability():
    module=umap_stub()
    module.model.graph_=csr_matrix(np.array([[0.,0.,0.],[0,0,1],[0,1,0]]))
    result=OperatorSnapshot(Y,dataset=SimpleNamespace(data=X),module=module)
    assert_array_equal(result['kernel'][0],np.zeros(3))
    assert_array_equal(result['affinity'][0],np.zeros(3))
