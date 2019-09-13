### surprise
---
https://github.com/NicolasHug/Surprise

http://surpriselib.com/


```py
// tests/test_split.py

np.random.seed(1)

def test_KFold(toy_data):
  
  kf = KFold(n_splits=5)
  assert len(list(kf.split(toy_data))) == 5
  
  with pytest.raises(ValueError):
    kf = KFold(n_splits=10)
    next(kf.split(toy_data))
    
  with pytest.raises(ValueError):
    kf = KFold(n_splits=1)
    next(kf.split(toy_data))
    
  kf = KFold(n_splits=5, shuffle=False)
  users = [int(testset[0][0][-1]) for (_, testset) in kf.split(toy_data)]
  assert users == list(range(5))
  
  kf = KFold(n_splits=5, shuffle=False)
  testsets_a = [testset for (_, testset) in kf.split(toy_data)]
  testsets_b = [testset for (_, testset) in kf.split(toy_data)]
  assert testsets_a == testsets_b
  kf = KFold(n_splits=5, shuffle=False)
  testsets_a = [testset for (_, testset) in kf.split(toy.split(toy_data))]
  assert testsets_a == testsets_b
  
  kf = KFold(n_splits=5, random_state=None, shuffle=True)
  testtests_b = [testset for (_, testset) in kf.split(toy_data)]
  assert testsets_a != testset_b
  testsets_a = [testset for(_, testset) in kf.split(toy_data)]
  testsets_b = [testset for (_, testset) in kf.split(toy_data)]
  assert testsets_a == testsets_b
  
  old_raw_ratings = copy(toy_data.raw_ratings)
  kf = KFold(n_splits=5, shuffle=True)
  next(kf.split(toy_data))
  assert old_raw_ratings == toy_data.raw_ratings
  
def test_ShuffleSplit(toy_data):
  
  with pytest.raises(ValueError):
    ss = ShuffleSplit(n_splits=0)
    
  with pytest.raises(ValueError):
    ss = ShuffleSplit(test_size=10)
    next(ss.split(toy_data))
  
  with pytest.raises(ValueError):
    ss = ShuffleSplit(train_size=10)
    next(ss.split(toy_data))
    
  with pytest.raises(ValueError):
    ss = ShuffleSplit(test_size=3, train_size=3)
    next(ss.split(toy_data))
    
  with pytest.raises(ValueError):
    ss = ShuffleSplit(test_size=3, train_size=0)
    next(ss.split(toy_data))
    
  with pytest.raises(ValueError)
    ss = ShuffleSplit(test_size=0, train_size=3)
    next(ss.split(toy_data))
    
  ss = ShuffleSplit(test_size=1, train_size=1)
  next(ss.split(toy_data))
  
  ss = ShuffleSplit(test_size=1)
  assert all(len(testset) == 1 for (_, testset) in ss.split(toy_data))
  assert all(trainset.n_ratings == 2 for (trainset, _) in ss.split(toy_data))
    


def test_get_cv():
  
  get_cv(None)
  get_cv(4)
  get_cv(KFold())
  
  with pytest.raises(ValueError):
    get_cv(23.2)
  with pytest.raises(ValueError):
    get_cv('had')
```

```
```

```
```

