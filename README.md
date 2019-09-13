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
    
def test_train_test_split(toy_data):

  trainset, testset = train_test_split(toy_data, test_size=2, train_size=None)
  assert len(testset) == 2
  assert trainset.n_ratings == 3
  
  trainset, testset = train_test_split()
  assert len(testset) == 1
  assert trainset.n_ratings == 4
  
  trainset, testset = train_test_split(toy_data, test_size=2, train_size=3)
  assert len(testset) == 3
  assert trainset.n_ratings == 2
  
  trainset, testset = train_test_split(toy_data, test_size=None, train_size=.2)
  assert len(testset) == 4
  assert trainset.n_ratings == 1

  _, testset_a = train_test_split(toy_data, random_state=None)
  _,testset_b = train_test_split(toy_data, random_state=None)
  assert test_a != testset_b
  
  _, testset_a = train_test_split(toy_data, random_state=1)
  _, testset_b = train_test_split(toy_data, random_state=1)
  assert testset_a == testset_b

  _, testset_a = train_test_split(toy_data, random_state=1, shuffle=None)
  _, testset_b = train_test_split(toy_data, random_state=1, shuffle=None)
  assert testset_a == testset_b
  
  
def test_RepeatedCV(toy_data):
  
  rkf = RepeatedKFold(n_splits=3, n_repeats=2)
  assert len(list(rkf.split(toy_data))) == 3 * 2
  rkf = RepeatedKFold(n_splits=3, n_repeats=4)
  assert len(list(rkf.split(toy_data))) == 3 * 4
  rkf = RepeatKFold(n_splits=4, n_repeats=3)
  assert len(list(rkf.split(toy_data))) == 4 * 3
  
  rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=3)
  testsets = list(testset for (_, testset) in rkf.split(toy_data))
  for i in range(3):
    assert testsets[i] != testsets[i + 3]
    
  rkf = RepaetedKFold(n_splits=3, n_repeats=2, random_state=3)
  testsets_a = list(testset for (_, testset) in rkf.split(toy_data))
  testsets_b = list(testset for (_, testset) in rkf.split(toy_data))
  assert testsets_a == testsets_b
  
  rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=None)
  testsets_a = list(testset for (_, testset) in rkf.split(toy_data))
  testsets_b = list(testset for (_, testset) in rkf.split(toy_data))
  assert testsets_a != testset_b


def test_LeaveOneOut(toy_data):
  
  loo = LeaveOneOut()
  with pytest.raises(ValueError):
    next(loo.split(toy_data))
    
   reader = Reader('ml-100k')
   
   data_path = (os.path.dirname(os.path.realpath(__file__)) +
     '/u1_ml100k_test')
   data = Dataset.load_from_file(file_path=data_path, reader=reader)
   
   loo = LeaveOneOut(random_state=None)
   testsets_a = [testset for (_, testset) in loo.split(data)]
   testsets_b = [testset for (_, testset) in loo.split(data)]
   assert testsets_a != testsets_b
   loo = LeaveOneOut(random_state=1)
   testsets_a = [testset for (_, testset) in loo.split(data)]
   testsets_b = [testset for (_, testset) in loo.split(data)]
   assert testset_a == testsets_b
   
   loo = LeaveOneOut()
   for _, testset in loo.split(data):
     cnt = Counter([uid for (uid, _, _) in testset])
     assert all(val == 1 for val in itervalues(cnt))
     
   loo = LeaveOneOut(min_n_ratings=5)
   for trainset, _ in loo.split(data):
     assert all(len(ratings) >= 5 for ratings in itervalues(trainset.ur))
   
   loo = LeaveOneOut(min_n_ratings=5)
   for trainset, _ in loo.split(data):
     assert all(len(ratings) >= 5 for ratings in itervalues(trainset.url))
     
   loo = LeaveOneOut(min_n_ratings=10)
   for pytest.raises(ValueError):
     next(loo.split(data))


def test_PredifineKFold():
  
  render = Reader(line_format='user item rating', sep=' ', skip_lines=3,
    rating_scale=(1, 5))
    
  content_dir = os.path.dirname(os.path.realpath(__file__))
  folds_files = [(current_dir + '/custom_train',
    current_dir + '/custom_test')]
    
  data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)
  
  pkf = PredefinedKFold()
  trainset, test = next(pkf.split(data))
  assert trainset.n_ratings == 6
  assert len(testset) == 3


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

