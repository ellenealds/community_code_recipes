## Cost Estimation Functions

This function takes an input string and outputs an estimate cost for that string. Cohere charges based on generation units, where one generation unit is up to 1k characters. Consider using this function to estimate previously run prompts and output costs, or to process multiple prompts to find the most cost effective one.

### `str_to_cost`

This function takes a string and model type specification and outputs an estimated cost for processing that string with the corresponding generative endpoint.

### Parameters

* `prompt` (`str`): A string that may be passed to co.generate
* `model` (`str`): Default or Custom model type

### Returns

* `float`: An estimate of cost in dollars for the prompt to be processed by the given model type.

### Example Usage

```python
>>> import math

>>> def str_to_cost(prompt, model="default"):
...     prompt_chars = len(prompt)
...     num_units = math.ceil(prompt_chars/1000)
...     if model == "custom":
...         return num_units * (5/1000)
...     else:
...         return num_units * (2.5/1000)

>>> str_to_cost("Write me a poem about dogs")
0.0025

```