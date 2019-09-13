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
    


```

```
```

```
```

