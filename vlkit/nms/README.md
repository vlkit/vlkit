<h1 align="center">Non Maximal Suppression</h1>

<p align="center">
  <img src="http://data.kaizhao.net/projects/vlkit/nms/nms-results.png" width=600px>
</p>

```
from vlkit import nms
I = np.array(Image.open("image.jpg"))
edge = np.array(Image.open("edge.png"))
edge1 = nms.nms(E)
```

See [test_nms.py](../../test/test_nms.py) for examples.
