shader_type particles;

uniform sampler2D fireflies_data;

void process() {
	ivec2 image_size = textureSize(fireflies_data, 0);
	
	ivec2 texel_pos = ivec2(int(mod(float(INDEX),float(image_size.x))), int(float(INDEX)/float(image_size.x)));
	vec4 data = texelFetch(fireflies_data, texel_pos, 0);
	
	TRANSFORM[3].xy = vec2(data.r, data.g);
	COLOR.a = data.b;
}
