#src # This is needed to make this run as normal Julia file:
using Markdown #src

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Get the packages
"""
using Pkg; Pkg.instantiate()

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# Geodata in Julia

The geodata ecosystem in Julia has matured a lot, but is not in a fully stable state yet.
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
A brief overview of a problematic situation:
- [YAXArrays.jl](https://github.com/JuliaDataCubes/YAXArrays.jl) named multidimensional arrays
- [Raster.jl](https://github.com/rafaqz/Rasters.jl) for raster data (geotiff, Netcdf, ascii-grid, etc)
- [DimensionalData.jl](https://github.com/rafaqz/DimensionalData.jl) shared backend for Rasters.jl and YAXArrays.jl
- [Shapefile.jl](https://github.com/JuliaGeo/Shapefile.jl) for, you guessed, shapefiles
- [ArchGDAL.jl](https://github.com/yeesian/ArchGDAL.jl) for interactions with the GDAL lib
- [Proj4.jl](https://github.com/JuliaGeo/Proj.jl) for map projections
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""

- https://juliageo.org/ -- biggest geo-group
- https://github.com/JuliaEarth -- for geostatistics
- https://ecojulia.org/ -- (spatial)ecology
- https://github.com/GenericMappingTools/GMT.jl (for Huw)
"""


#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Named multidimensional data

This tutorial has been adapted from various sources such as:
- YAXArrays docs
- https://github.com/JuliaDataCubes/datacubes_in_julia_workshop

Let's start to get aquainted with YAXArrays:
"""
using YAXArrays

yaxa_rand = YAXArray(rand(5, 10))

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
So far nothing magical has happened, why do we need a whole new package?
If we want to create a `YAXArray` with named dimension we need the following:
- axes or dimensions with names and tick values

"""
using DimensionalData

axlist = (
    Dim{:time}(range(1, 20, length=20)),
    X(range(1, 10, length=10)),
    Y(range(1, 5, length=15)),
    Dim{:variable}(["temperature", "precipitation"])
)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
- the data to feed to the `YAXArray` matching the dimensions defined in the axlist:
"""
data = rand(20, 10, 15, 2)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
- and additionally some metadata:
"""
props = Dict(
    "origin" => "YAXArrays.jl example",
    "x" => "longitude",
    "y" => "latitude",
);

a2 = YAXArray(axlist, data, props)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
We can now access our `YAXArray` at any variable or time point we want:
"""
a2[variable=At("temperature"), time=1].data

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Reading data

In order to read specific data types (Zarr, NetCDF, etc...) we need to load the necessary backend first:

"""
using Zarr

bucket = "esdl-esdc-v3.0.2"
store = "esdc-16d-2.5deg-46x72x1440-3.0.2.zarr"
path = "https://s3.bgc-jena.mpg.de:9000/" * bucket * "/" * store
c = Cube(open_dataset(zopen(path,consolidated=true,fill_as_missing=true)))

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
### Operations on data

Like normal arrays you can modify the data performing simple arithmetics. Additional operations can be performed in different levels of complexity:
- using `map`: applies functions to each element of an array
- using `mapslices`: reduce dimensions
- using `mapCube`: applies functions to an array that may change dimensions

"""
yaxa_rand.data
#-
add_yaxa = yaxa_rand .+ 5
add_yaxa.data

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Let's apply more complex functions with `map` to each element of the array individually:
"""
offset = 5
map(a2) do x
    (x + offset) / 2 * 3
end

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
This computation happens lazily, allowing operations to run fast.

Now let's see how we can apply external functions to the data:

"""
import Statistics: mean
a2_timemean = mapslices(mean, a2, dims="Time")

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
We computed the average value of the points in time, so no time variable is present in the final cube.
Similartly we can compute the spatial means in one value per time step:

"""
a2_spacemean = mapslices(mean, a2, dims=("X", "Y"))

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Now to the most flexible way to apply a function: `mapCube`. With it you can directly modify dimension, adding or removing them

Let's use the esdc and apply the median function:

"""
using Statistics 
indims = InDims("time")
outdims = OutDims()
function apply_median(xout, xin)
    x = filter(!ismissing, xin)
    x = filter(!isnan,x)

    xout[] = isempty(x) ? missing : median(x)
end

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
See how the user defined function passed to mapCube has to have the signature f(outputs..., inputs...) and potentially followd by additional arguments and keyword args.

"""
medians = mapCube(apply_median, c[Variable=Where(contains("temp"))];indims, outdims)


#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# Exercise

- Load "bands.zarr" 
- Define some spectral indices calulations for your favourite indices. Keep in mind that we have the following bands:
  - B02 (Band 2): Blue (approximately 490 nm)
  - B03 (Band 3): Green (approximately 560 nm)
  - B04 (Band 4): Red (approximately 665 nm)
  - B08 (Band 8): Near Infrared (NIR) (approximately 842 nm)
The results are given leveraging the NDVI = (N-R)/(N+R)
- If you use NDVI you have to divide the data by 10000 to get it in the expected range.
- Apply this calculations to the data in the cube. Use a naive approach, `map` and `mapCube`. Are the results equivalent?
"""