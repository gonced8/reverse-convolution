### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 9ba38cfe-f46d-11ea-039b-2d11eb144c98
begin
	import Pkg
	Pkg.activate(mktempdir())
end

# ╔═╡ 88cfc46c-f46d-11ea-21df-59c027e4b86b
begin
	Pkg.add(["Images", "ImageIO", "ImageMagick", "ImageTransformations", "ImageFiltering"])
	using Images, LinearAlgebra, SparseArrays
end

# ╔═╡ a3f814ac-f46c-11ea-33d7-81d4c435aba2
begin
	large_image = load("img1.jpg")
	image = imresize(large_image; ratio=0.1)
	#image = Gray.(image)
	#image = Gray.(rand(Float64, (5, 5)))
end

# ╔═╡ e731fd9c-f475-11ea-0123-b5df0503c9ba
size(image)

# ╔═╡ 5965dbc0-f4fa-11ea-3d29-536288bec4f6
typeof(image)

# ╔═╡ 2d535372-f479-11ea-0395-1d9e0339bfde
function Kernel_uniform(shape)
	value = 1/prod(shape)
	return fill(value, shape)
end

# ╔═╡ 820976ec-f476-11ea-0113-d909496fa398
begin
	#kernel = Kernel_uniform((7, 7))
	kernel = Kernel.gaussian((3, 3))
	size(kernel)
end

# ╔═╡ ece7034c-f5b8-11ea-374f-c7609880175a
function get_pad(kernel)
	return (size(kernel).-1).÷2
end

# ╔═╡ 9017eb36-f47d-11ea-0ae8-f3b3bb85bf53
begin
	pad = Pad(:replicate, get_pad(kernel))
	#pad = Fill(0, size(kernel))
	filtered_pad = imfilter(image, kernel, pad, Algorithm.FIR())
end

# ╔═╡ d89f858a-f47d-11ea-20fc-ed418cc5653b
size(filtered_pad)

# ╔═╡ a636b05c-f566-11ea-1e0b-0515f16c7884
#=
begin
	function get_b(img, f, pad, mat, idx)
		pad_size = get_pad(f)

		if pad=="replicate"
			img_padded = padarray(img, Pad(:replicate,pad_size))
			img_padded[1:end-pad_size[1], 1:end-pad_size[2]] .= 0
		else
			return 0
		end

		mono = (length(size(channelview(img)))==2)

		if mono
			b = mat*vec(img)
			return reshape(b[idx], size(img))
		else
			channels = channelview(img)
			b = zeros(size(img))
			for i=1:3
				b_i = mat*vec(channels[i, :, :])
				b[i, :, :] = reshape(b_i[idx], size(img))
			end
			return colorview(RGB, b)
		end
	end

	function f2mat2(img, f, pad="zero")
		# https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication

		pad_size = get_pad(f)
		mask = padarray(fill(true, size(img)), Fill(false, pad_size))

		input_shape = size(mask)
		output_shape = input_shape .- size(f) .+ (1, 1)
		input_size = prod(input_shape)
		output_size = prod(output_shape)

		mat = sparse([], [], Float64[], input_size, output_size)

		delta = input_shape[1]-size(f, 1)
		f_flat = sparse([], [], Float64[], input_shape[1], size(f, 2))

		f_flat[1:size(f, 1), :] = parent(f)
		f_flat = sparsevec(f_flat)[1:end-delta]

		delta = input_shape[1] - size(f, 1) + 1

		for i=1:size(mat, 2)÷delta, j=1:delta
			ii = (i-1)*input_shape[1]+j
			jj = (i-1)*delta+j
			mat[ii:ii+length(f_flat)-1, jj] = f_flat
		end

		mask_flat = vec(mask)
		#b = get_b(img, f, pad, mat, mask_flat)

		idx = [i for i=1:length(mask) if mask[i]]
		mat = mat[idx, :]

		return mat'
	end
end
=#

# ╔═╡ 64d41c96-f569-11ea-28d7-b9f8ef35d8db
function f2mat(img, f, pad=Pad(:replicate, get_pad(kernel)))
	input_shape = size(img)
	input_size = prod(input_shape)

	indices = reshape(1:input_size, input_shape)

	pad_size = get_pad(f)
	indices = padarray(indices, pad)

	indices = parent(indices)
	f = parent(f)
	
	output_shape = size(indices) .- size(f) .+ (1, 1)
	output_size = prod(output_shape)
	
	mat = sparse([], [], Float64[], input_size, output_size)
	
	for j=1:output_shape[2], i=1:output_shape[1]
		for n=1:size(f, 2), m=1:size(f, 1)
			input_idx = indices[i+m-1, j+n-1]
			if input_idx > 0
				output_idx = i + (j-1)*output_shape[1]
				mat[input_idx, output_idx] += f[m, n]
			end
		end
	end
	
	return mat'
end

# ╔═╡ 3626ce80-f482-11ea-15f0-497b0ee9b3a4
#matrix = f2mat(image, kernel, "zero")

# ╔═╡ 1a26054e-f4f9-11ea-3faa-23a1ad8febd0
function convolution(image, kernel, pad=Pad(:replicate, get_pad(kernel)))
	matrix = f2mat(image, kernel, pad)
	mono = (length(size(channelview(image)))==2)
	
	if mono
		return reshape(matrix*vec(image), size(image))
	else
		channels = channelview(image)
		new_image = [reshape(matrix*vec(channels[i, :, :]), size(image)) for i=1:3]
		return colorview(RGB, new_image[1], new_image[2], new_image[3])
	end
end

# ╔═╡ 30f93266-f4fa-11ea-35d5-b7bd56cb73e3
#convolution(image, kernel, "replicate")

# ╔═╡ e2a404cc-f511-11ea-3c9d-d31aa63b143d


# ╔═╡ 63100174-f512-11ea-3d6b-39684a9ca980
function get_original(image, kernel, pad=Pad(:replicate, get_pad(kernel)))
	matrix = f2mat(image, kernel, pad)
	
	mono = (length(size(channelview(image)))==2)
	
	if mono
		channel = channelview(image)
		return colorview(Gray, reshape(matrix\vec(channel), size(image)))
	else
		channels = channelview(image)
		new_image = zeros(size(channels))
		for i=1:3
			new_image[i, :, :] = matrix\vec(channels[i, :, :])
		end
		return colorview(RGB, new_image)
	end
end

# ╔═╡ 4bd86730-f512-11ea-1c03-bfd6d008e37a
begin
	estimate = false
	
	if !estimate
		estimated_kernel = kernel
		estimated_pad = pad
	else
		estimated_kernel = Kernel.gaussian((3, 3))
		#estimated_kernel = Kernel_uniform((7, 7))
		#estimated_pad = Fill(0, get_pad(estimated_kernel))
		estimated_pad = Pad(:replicate, get_pad(estimated_kernel))
	end
	
	estimated = get_original(filtered_pad, estimated_kernel, estimated_pad)
end

# ╔═╡ 4eb55d58-f514-11ea-064c-2b8a08b3c1eb
[image filtered_pad estimated]

# ╔═╡ Cell order:
# ╠═9ba38cfe-f46d-11ea-039b-2d11eb144c98
# ╠═88cfc46c-f46d-11ea-21df-59c027e4b86b
# ╠═a3f814ac-f46c-11ea-33d7-81d4c435aba2
# ╠═e731fd9c-f475-11ea-0123-b5df0503c9ba
# ╠═5965dbc0-f4fa-11ea-3d29-536288bec4f6
# ╠═2d535372-f479-11ea-0395-1d9e0339bfde
# ╠═820976ec-f476-11ea-0113-d909496fa398
# ╠═ece7034c-f5b8-11ea-374f-c7609880175a
# ╠═9017eb36-f47d-11ea-0ae8-f3b3bb85bf53
# ╠═d89f858a-f47d-11ea-20fc-ed418cc5653b
# ╠═a636b05c-f566-11ea-1e0b-0515f16c7884
# ╠═64d41c96-f569-11ea-28d7-b9f8ef35d8db
# ╠═3626ce80-f482-11ea-15f0-497b0ee9b3a4
# ╠═1a26054e-f4f9-11ea-3faa-23a1ad8febd0
# ╠═30f93266-f4fa-11ea-35d5-b7bd56cb73e3
# ╟─e2a404cc-f511-11ea-3c9d-d31aa63b143d
# ╠═63100174-f512-11ea-3d6b-39684a9ca980
# ╠═4bd86730-f512-11ea-1c03-bfd6d008e37a
# ╠═4eb55d58-f514-11ea-064c-2b8a08b3c1eb
