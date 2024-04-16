
(
echo "from collections import defaultdict"
echo "import matplotlib.pyplot as plt"
echo "import numpy as np"
echo "data = defaultdict(lambda: defaultdict(float))"
./view_stats.sh | awk '{print "data[",$1,"][",$2,"]= ",$3}' | grep -v '====' | sed 's/ //g' | grep -vi GENERALI

echo "for test in [False, True]:"
echo " plt.clf()"
echo " for d in data.keys():"
echo "  if ('TEST' in d) == test:"
echo "   x = [x_ for x_ in data[d].keys()]"
echo "   x = sorted(x)"
echo "   y = [float(data[d] [x_]) if x_ > 0 else (np.average([float(data[dp] [x_]) for dp in data.keys() if (test == ('TEST' in dp)) ])) for x_ in x]"
echo "   plt.plot(x, y, label=d)"
echo "   plt.text(x[-1],y[-1],d)"
echo " plt.tight_layout()"
echo " plt.legend()"
echo " plt.savefig(f'test{test}_result.png')"

) > plotter.py
sed  -i.prout 's/\]\[/\"&/g' plotter.py
cat plotter.py |  tr -cd '\11\12\15\40-\176' > newplotter.py

python newplotter.py
open test*result.png
