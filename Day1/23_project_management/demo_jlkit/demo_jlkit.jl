### A Pluto.jl notebook ###

using Markdown
using InteractiveUtils

# ╔═╡ 307335ba-cf46-11eb-28b0-f199cf048ae6
md" ## Your title here"

# ╔═╡ dfdfa33f-53c1-45e2-8727-86ac01a09398
md"""

## Brief intro to Pluto notebooks and Markdown syntax

Feel free to skip this tutorial if you already know how Pluto noteboos work. Also, make sure to delete it once you submit/share your notebook.

This is your notebook. When filling it, you should obey Pluto's own syntax to create cells containing code or text.

To write one single line of narrative text, use simple quotes and include "md" before the quotes (as done for the title header above). To write several lines of text, use three pairs of quotes, as in this cell.

Text written with no special markings will appear without any special formatting when you generate a `.html` or `.pdf` version of this file (click the "Export" button above to convert it).

It is possible to include code inside Markdown cells (it is not executed, though):

- either as `inline code`

- or as a block of code:

```{julia}
print("This is an example of a code block...")
print("for several lines of code")
```

It is also possible to include hyperlinks: for example, this [link to further details on the Markdown syntax used in Pluto notebooks](https://www.juliapackages.com/p/pluto).

Cells for code do not need any special marking, unless the cell contains several lines of code. In that case, the code should be included inside a `begin ... end` block.

```{julia}
begin
    # your lines
    # of code
end
```

To execute the code in code cell and have its results appear above it, click the "play" ("Run cell") button below the cells or place your cursor inside the chunk and press `Shift+Enter`.

The outputs of a cell are always show, unless the cell is disabled (available in the `...` - "Actions" - button on top of the cell). To hide the contents of the cell itself, click the eye icon on top of the cell.
"""

# ╔═╡ 37c7bffc-cf46-11eb-19b8-4175e38e818b
begin
# We suggest having a chunk dedicated to variables containing the paths to the folders related to the project.
# The code in this chunk is not relevant for the reader, and thus is not included in the knitted version (therefore, `include = FALSE`).
# This is also useful if you are not using the folder structure suggested by this kit, and want to preserve your privacy.
# Feel free to edit these paths to adapt them to your needs or not use this suggestion at all.

	data_dir = joinpath("results", "data", "processed")  ## Do NOT play with stuff in data/raw. That is your back-up. Work only on `process`.
	scripts_dir = joinpath("scripts")
	figs_dir = joinpath("results", "figures")
	tabs_dir = joinpath("results", "tables")
	suppl_dir = joinpath("results", "supplementary")
end

# ╔═╡ f5062ac5-18e0-42fd-8bb7-1602a8298779
md"## Load data"

# ╔═╡ d4465a06-7d32-11ec-3907-59e1f4d60e2f
# Here, you write the code to read the data from data_dir

# ╔═╡ 6d256dbe-edc9-4132-aac7-62a473601034
md"""
## Data analysis

### Figure 1
"""

# ╔═╡ f12401f7-1c08-4800-a64d-88f91f6b59fe
# Here, you write the code to create a graph (Figure 1) included in the main text.
# To avoid the repetition of having the same figure here as in the text, include a line that saves the figure in the figures directory.

# ╔═╡ 2af762fe-0173-47eb-98d7-8fa6d3b555d3
md"### Figure S1"

# ╔═╡ 93c7698d-6e5b-4803-b7c6-90d4da7cd9bf
# your code to plot Figure S1

# ╔═╡ Cell order:
# ╠═307335ba-cf46-11eb-28b0-f199cf048ae6
# ╠═dfdfa33f-53c1-45e2-8727-86ac01a09398
# ╠═37c7bffc-cf46-11eb-19b8-4175e38e818b
# ╠═f5062ac5-18e0-42fd-8bb7-1602a8298779
# ╠═d4465a06-7d32-11ec-3907-59e1f4d60e2f
# ╠═6d256dbe-edc9-4132-aac7-62a473601034
# ╠═f12401f7-1c08-4800-a64d-88f91f6b59fe
# ╠═2af762fe-0173-47eb-98d7-8fa6d3b555d3
# ╠═93c7698d-6e5b-4803-b7c6-90d4da7cd9bf