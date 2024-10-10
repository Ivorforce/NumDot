shader_type canvas_item;

const vec3 ON_COLOR = vec3(0.9, 0.95, 0.99);
const float EDGE = 0.01;
const vec4 EDGE_COLOR = vec4(0.6, 0.6, 0.6, 1.0);

void fragment() {
	vec2 UV_INNER = (UV - EDGE) / (1.0 - EDGE * 2.0);
	if (UV_INNER.x < 0.0 || UV_INNER.x >= 1.0 || UV_INNER.y < 0.0 || UV_INNER.y >= 1.0) {
		COLOR = EDGE_COLOR;
	}
	else {
		COLOR = vec4(texture(TEXTURE, UV_INNER).r * 255.0 * ON_COLOR, 1.0);
	}
}
