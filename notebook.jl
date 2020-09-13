### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 88cfc46c-f46d-11ea-21df-59c027e4b86b
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add(["Images", "ImageIO", "ImageMagick", "ImageTransformations", "ImageFiltering"])
	using Images, LinearAlgebra, SparseArrays
	md"Libraries added"
end

# ╔═╡ 045fc73c-f5bb-11ea-210d-c3a88c7b5c08
md"# Reversing a convolution"

# ╔═╡ ce1dd8fc-f5bb-11ea-3cf4-4b20b1250bb1
md"## Functions"

# ╔═╡ f1cf67d8-f5bc-11ea-0015-192f0ebb1fdf
md"### Kernel"

# ╔═╡ f284c890-f5bb-11ea-30e8-abbc1f458cec
function Kernel_uniform(shape)
	value = 1/prod(shape)
	return fill(value, shape)
end

# ╔═╡ fcae7582-f5bb-11ea-2b79-43f8cd3178c3
function get_pad(kernel)
	return (size(kernel).-1).÷2
end

# ╔═╡ e6b903c2-f5bc-11ea-0c2c-03e099517d90
md"### Convolution and Reverse"

# ╔═╡ fb654aee-f5bc-11ea-35f4-3397194c8722
md"## Example"

# ╔═╡ 116fe702-f5bd-11ea-0ba9-c55263d026ba
md"### Importing image"

# ╔═╡ a3f814ac-f46c-11ea-33d7-81d4c435aba2
begin
	i = 2
	
	if i==1
		filename = "monkey"
		large_image = load("monkey.jpg")
		image = imresize(large_image; ratio=0.15)
	elseif i==2
		filename = "lena"
		large_image = load("lena.jpg")
		image = imresize(large_image; ratio=0.35)
	end
	
	#image = Gray.(image)
	#image = Gray.(rand(Float64, (5, 5)))
end

# ╔═╡ e731fd9c-f475-11ea-0123-b5df0503c9ba
size(image)

# ╔═╡ 5965dbc0-f4fa-11ea-3d29-536288bec4f6
typeof(image)

# ╔═╡ 1eb50ff0-f5bd-11ea-34a9-65ceeb74698a
md"### Initializing filter"

# ╔═╡ 820976ec-f476-11ea-0113-d909496fa398
begin
	#kernel = Kernel_uniform((3, 3))
	kernel = Kernel.gaussian((3, 3))
	
	pad = Pad(:replicate, get_pad(kernel))
	#pad = Fill(0, size(kernel))
	
	size(kernel)
end

# ╔═╡ 110e881e-f5bc-11ea-0a20-3dabe51098a3
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

# ╔═╡ 9966750a-f5bc-11ea-2402-f16ff519f56d
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

# ╔═╡ a053dae2-f5bc-11ea-11fb-7115b2489efa
function get_original(image, kernel, pad=Pad(:replicate, get_pad(kernel)))
	matrix = f2mat(image, kernel, pad)
	
	mono = (length(size(channelview(image)))==2)
	
	if mono
		channel = channelview(image)
		estim = matrix\vec(channel)
		estim = [(x>1) ? 1 : (x<0) ? 0 : x for x in estim]	# image values in [0, 1]
		return Gray.(reshape(estim, size(image)))
	else
		channels = channelview(image)
		new_image = zeros(size(channels))
		for i=1:3
			estim = matrix\vec(channels[i, :, :])
			estim = [(x>1) ? 1 : (x<0) ? 0 : x for x in estim] # in [0, 1]
			new_image[i, :, :] = estim
		end
		return colorview(RGB, new_image)
	end
end

# ╔═╡ 3d477c82-f5bd-11ea-2571-f5cdb9fc1525
md"### Filtered image"

# ╔═╡ 9017eb36-f47d-11ea-0ae8-f3b3bb85bf53
filtered_pad = imfilter(image, kernel, pad, Algorithm.FIR())

# ╔═╡ d89f858a-f47d-11ea-20fc-ed418cc5653b
size(filtered_pad)

# ╔═╡ 3626ce80-f482-11ea-15f0-497b0ee9b3a4
#matrix = f2mat(image, kernel, "zero")

# ╔═╡ 30f93266-f4fa-11ea-35d5-b7bd56cb73e3
#convolution(image, kernel, "replicate")

# ╔═╡ 52ae79ae-f5bd-11ea-1d9b-a5963c9bcd16
md"### Reversing the convolution"

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
	
	recovered = get_original(filtered_pad, estimated_kernel, estimated_pad)
end

# ╔═╡ 68b18282-f5bd-11ea-1801-933fb61400ab
md"#### Original | Filtered | Recovered"

# ╔═╡ 4eb55d58-f514-11ea-064c-2b8a08b3c1eb
[image filtered_pad recovered]

# ╔═╡ 18642c9c-f5bf-11ea-2d7f-0956bb770cc7
begin
	save(filename * "_original.jpg", image)
	save(filename * "_blurred.jpg", filtered_pad)
	save(filename * "_recovered.jpg", recovered)
	md"Files saved"
end

# ╔═╡ d6714f18-f60b-11ea-15b6-517d7006d9e4
md"old functions inside"
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

# ╔═╡ Cell order:
# ╟─045fc73c-f5bb-11ea-210d-c3a88c7b5c08
# ╟─ce1dd8fc-f5bb-11ea-3cf4-4b20b1250bb1
# ╟─88cfc46c-f46d-11ea-21df-59c027e4b86b
# ╟─f1cf67d8-f5bc-11ea-0015-192f0ebb1fdf
# ╟─f284c890-f5bb-11ea-30e8-abbc1f458cec
# ╟─fcae7582-f5bb-11ea-2b79-43f8cd3178c3
# ╟─e6b903c2-f5bc-11ea-0c2c-03e099517d90
# ╟─110e881e-f5bc-11ea-0a20-3dabe51098a3
# ╟─9966750a-f5bc-11ea-2402-f16ff519f56d
# ╟─a053dae2-f5bc-11ea-11fb-7115b2489efa
# ╟─fb654aee-f5bc-11ea-35f4-3397194c8722
# ╟─116fe702-f5bd-11ea-0ba9-c55263d026ba
# ╠═a3f814ac-f46c-11ea-33d7-81d4c435aba2
# ╟─e731fd9c-f475-11ea-0123-b5df0503c9ba
# ╟─5965dbc0-f4fa-11ea-3d29-536288bec4f6
# ╟─1eb50ff0-f5bd-11ea-34a9-65ceeb74698a
# ╠═820976ec-f476-11ea-0113-d909496fa398
# ╟─3d477c82-f5bd-11ea-2571-f5cdb9fc1525
# ╠═9017eb36-f47d-11ea-0ae8-f3b3bb85bf53
# ╟─d89f858a-f47d-11ea-20fc-ed418cc5653b
# ╠═3626ce80-f482-11ea-15f0-497b0ee9b3a4
# ╠═30f93266-f4fa-11ea-35d5-b7bd56cb73e3
# ╟─52ae79ae-f5bd-11ea-1d9b-a5963c9bcd16
# ╠═4bd86730-f512-11ea-1c03-bfd6d008e37a
# ╟─68b18282-f5bd-11ea-1801-933fb61400ab
# ╠═4eb55d58-f514-11ea-064c-2b8a08b3c1eb
# ╟─18642c9c-f5bf-11ea-2d7f-0956bb770cc7
# ╟─d6714f18-f60b-11ea-15b6-517d7006d9e4
