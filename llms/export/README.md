# Export LLMs to C++

Export language model inference from Python to run directly in C++.

To run, first install the requirements:

```bash
pip install -U mlx-lm
```

Then generate text from Python with:

```bash
python export.py generate "How tall is K2?"
```

To export the generation function run:

```bash
python export.py export
```

Then build the C++ code (requires CMake):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

And run the generation from C++ with:

```bash
./build/main lama3.1-instruct-4bit "How tall is K2?"
```
