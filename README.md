# color_wheel
Creates a color wheel and adds symbols to it at RGB locations

## Description
This utility generates an RGB color wheel that attempts to include the full color space. Once it's generated, it adds symbols to the color wheel at color location specified in a CSV file. It can use a few different symbols, keyed on the value in one column of the CSV file.

### Input data format
The program assumes that the RGB color values that represent locations for each point are stored in columns named R, G, and B.

## Requirements
Requires pip to be installed, as well as `pypng`, `numpy`, and `scikit-image`.

## Usage
`./color_wheel.py <infile> <outfile>`

Further options are documented in the script, and can be viewed with `./color_wheel.py -h`