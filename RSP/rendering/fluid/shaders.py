"""GLSL passes; all scene inputs are GPU textures, never host images."""
QUAD = '''#version 330 core
out vec2 uv;
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p;gl_Position=vec4(p*2.-1.,0,1);}
'''
PARTICLE = '''#version 330 core
layout(location=0) in vec2 corner;
layout(location=1) in vec4 center_radius;
layout(location=2) in vec4 color;
uniform mat4 view, projection;
uniform float radius_scale;
out vec2 local;
out vec3 eye;
out vec4 tint;
out float radius;
void main(){local=corner;radius=center_radius.w*radius_scale;
eye=(view*vec4(center_radius.xyz,1)).xyz;
gl_Position=projection*vec4(eye+vec3(corner*radius,0),1);tint=color;}
'''
COMMON = '''
uniform sampler2D scene_depth;
uniform mat4 inv_projection, projection;
uniform vec2 size;
vec3 position(vec2 p,float d){vec4 v=inv_projection*vec4(p*2.-1.,d*2.-1.,1);return v.xyz/v.w;}
float sceneZ(vec2 p){return -position(p,texture(scene_depth,p).r).z;}
'''
TRANSPARENT = '#version 330 core\n' + COMMON + '''
in vec2 local;in vec3 eye;in vec4 tint;in float radius;
layout(location=0) out vec4 accumulation;
layout(location=1) out float reveal;
layout(location=2) out float weighted_depth;
uniform bool depth_enabled, layer_blending;
void main(){float z=-eye.z;float gap=sceneZ(gl_FragCoord.xy/size)-z;if(gap<=0.)discard;
float a=tint.a;
if(!layer_blending){
    // Depth testing selects one square face; transmit the background once.
    accumulation=vec4(tint.rgb*a,a);reveal=1.-a;
    weighted_depth=depth_enabled?z*a:0.;return;
}
float d=z/10.;float w=a*clamp(10./(1e-5+pow(2.*d,2.)+pow(.6*d,6.)),.01,300.);
accumulation=vec4(tint.rgb*a,a)*w;reveal=a;
weighted_depth=depth_enabled?z*a*w:0.;}
'''
RESOLVE = '''#version 330 core
in vec2 uv;uniform sampler2D background, accumulation, reveal;
uniform sampler2D weighted_depth, depth_range;
uniform bool depth_enabled;
uniform float depth_strength;
out vec4 result;
void main(){vec4 a=texture(accumulation,uv);float t=clamp(texture(reveal,uv).r,0.,1.);
vec3 color=a.rgb/max(a.a,1e-5);
if(depth_enabled && a.a>1e-5){
    vec2 range=texelFetch(depth_range,ivec2(0),0).rg;
    float z=texture(weighted_depth,uv).r/a.a;
    // Nearly flat surfaces have no useful depth contrast; do not amplify noise.
    if(range.y-range.x>max(.01,abs(range.x)*.001)){
        float d=smoothstep(range.x,range.y,z);
        color*=mix(1.+.1*depth_strength,1.-.5*depth_strength,d);
    }
}
result=vec4(mix(color,texture(background,uv).rgb,t),1);}
'''
DEPTH_RANGE = '''#version 330 core
uniform sampler2D source, accumulation;
uniform bool first_level;
out vec2 result;
void main(){
    ivec2 size=textureSize(source,0), origin=ivec2(gl_FragCoord.xy)*2;
    vec2 bounds=vec2(1e20,-1e20);
    for(int y=0;y<2;y++)for(int x=0;x<2;x++){
        ivec2 p=origin+ivec2(x,y);
        if(any(greaterThanEqual(p,size)))continue;
        vec2 sample_range;
        if(first_level){
            float weight=texelFetch(accumulation,p,0).a;
            if(weight<=1e-5)continue;
            sample_range=vec2(texelFetch(source,p,0).r/weight);
        }else sample_range=texelFetch(source,p,0).rg;
        bounds=vec2(min(bounds.x,sample_range.x),max(bounds.y,sample_range.y));
    }
    result=bounds;
}
'''
DEPTH = '#version 330 core\n' + COMMON + '''
in vec2 local;in vec3 eye;in vec4 tint;in float radius;
layout(location=0) out vec2 result;
void main(){float r=dot(local,local);if(r>1.)discard;
float z=-eye.z-radius*sqrt(1.-r);if(z<=0.||z>=sceneZ(gl_FragCoord.xy/size))discard;
result=vec2(z,radius);}
'''
THICKNESS = '#version 330 core\n' + COMMON + '''
in vec2 local;in vec3 eye;in vec4 tint;in float radius;
layout(location=0) out float result;
#ifdef VELOCITY
uniform sampler2D nearest_depth;
uniform vec2 alpha_range;
uniform float attribute_depth_range;
layout(location=1) out vec4 color_weight;
layout(location=2) out float coverage_weight;
#endif
void main(){float r=dot(local,local);if(r>1.)discard;float h=radius*sqrt(1.-r);
float front=-eye.z-h;float back=min(-eye.z+h,sceneZ(gl_FragCoord.xy/size));
result=max(0.,back-max(front,0.));
#ifdef VELOCITY
color_weight=vec4(0.);
coverage_weight=0.;
float nearest=texture(nearest_depth,gl_FragCoord.xy/size).r;
float delta=max(0.,front-nearest)/max(radius,1e-6);
if(result>0. && delta<=attribute_depth_range){
    float weight=(1.-r)*exp(-.5*delta*delta);
    // The existing GPU packer stores alpha = low + (high-low)*smoothstep(speed).
    // Recover that same coverage without another velocity buffer or upload.
    float coverage=clamp((tint.a-alpha_range.x)/(alpha_range.y-alpha_range.x),0.,1.);
    color_weight=vec4(tint.rgb*weight,weight);
    coverage_weight=coverage*weight;
}
#endif
}
'''
THICKNESS_VELOCITY = THICKNESS.replace('#version 330 core\n', '#version 330 core\n#define VELOCITY\n', 1)
BLUR = '''#version 330 core
in vec2 uv;out vec2 result;uniform sampler2D source;
uniform vec2 size, axis;uniform float focal;uniform int max_radius;
void main(){vec2 c=texture(source,uv).rg;if(c.x>1e5){result=c;return;}
float range=clamp(2.*c.y*focal/c.x,1.,float(max_radius));
float sum=0.,weight=0.;
for(int i=-12;i<=12;i++){if(abs(float(i))>range)continue;
vec2 n=texture(source,uv+axis*float(i)/size).rg;
float dz=(n.x-c.x)/max(2.*c.y,1e-5);if(abs(dz)>3.)continue;
float w=exp(-2.*float(i*i)/(range*range)-.5*dz*dz);sum+=n.x*w;weight+=w;}
result=vec2(sum/max(weight,1e-6),c.y);}
'''
SURFACE = '#version 330 core\n' + COMMON + '''
in vec2 uv;out vec4 result;
uniform sampler2D background, surface, thickness, environment;
uniform bool has_environment;
uniform mat4 inv_view;
uniform vec3 sky_up, sky_down, sun, absorption;
uniform vec3 base_color;
#ifdef VELOCITY
uniform sampler2D material_color, material_coverage;
uniform float white_opacity;
#else
uniform bool billboard_blue;
#endif
uniform float ior;
vec3 eyePoint(vec2 p,float z){vec3 ray=position(p,.5);return ray*(z/max(-ray.z,1e-6));}
void main(){vec3 bg=texture(background,uv).rgb;float z=texture(surface,uv).r;
if(z>1e5||z>=sceneZ(uv)){result=vec4(bg,1);return;}
vec2 texel=1./size;vec3 p=eyePoint(uv,z);
float zl=texture(surface,uv-vec2(texel.x,0)).r,zr=texture(surface,uv+vec2(texel.x,0)).r;
float zb=texture(surface,uv-vec2(0,texel.y)).r,zt=texture(surface,uv+vec2(0,texel.y)).r;
vec3 dx=abs(zr-z)<abs(z-zl)?eyePoint(uv+vec2(texel.x,0),zr)-p:p-eyePoint(uv-vec2(texel.x,0),zl);
vec3 dy=abs(zt-z)<abs(z-zb)?eyePoint(uv+vec2(0,texel.y),zt)-p:p-eyePoint(uv-vec2(0,texel.y),zb);
vec3 n=normalize(cross(dx,dy));vec3 v=normalize(-p);if(dot(n,v)<0.)n=-n;
float t=clamp(texture(thickness,uv).r,0.,10.);
vec3 refr=refract(-v,n,1./ior);vec4 q=projection*vec4(p+refr*min(t,1.),1);
vec2 sampleUV=q.xy/q.w*.5+.5;
if(any(lessThan(sampleUV,vec2(0)))||any(greaterThan(sampleUV,vec2(1)))||sceneZ(sampleUV)<z)sampleUV=uv;
vec3 trans=exp(-absorption*t);
#ifdef VELOCITY
// Selective RGB absorption shifts pale speed colors towards orange. Keep
// thickness attenuation, but make it neutral for this visualization preset.
trans=vec3(exp(-dot(absorption,vec3(1./3.))*t));
#else
if(billboard_blue)trans=vec3(exp(-dot(absorption,vec3(1./3.))*t));
#endif
vec3 color=base_color;
float coverage=0.;
#ifdef VELOCITY
vec4 weighted=texture(material_color,uv);
if(weighted.a>1e-6){
    color=weighted.rgb/weighted.a;
    coverage=clamp(texture(material_coverage,uv).r/weighted.a,0.,1.);
}
#endif
vec3 refracted_background=texture(background,sampleUV).rgb;
vec3 transmitted=refracted_background*trans+color*(1.-trans);
vec3 reflected=normalize(mat3(inv_view)*reflect(-v,n));
vec3 env=mix(sky_down,sky_up,smoothstep(-.2,.7,reflected.z));
// Match Newton's Z-up equirectangular environment orientation.
if(has_environment){vec2 e=vec2(atan(-reflected.y,reflected.x)/6.2831853+.5,asin(clamp(reflected.z,-1.,1.))/3.1415927+.5);env=texture(environment,e).rgb;}
vec3 worldN=normalize(mat3(inv_view)*n),worldV=normalize(mat3(inv_view)*v);
float spec=pow(max(dot(worldN,normalize(sun+worldV)),0.),100.);
float f0=pow((ior-1.)/(ior+1.),2.);float f=f0+(1.-f0)*pow(1.-max(dot(n,v),0.),5.);
vec3 water=mix(transmitted,env,f)+vec3(spec*.35);
#ifdef VELOCITY
// The fast endpoint retains background transmission even in thick fluid.
// Scaling coverage alone would leave opaque Water underneath it.
color=mix(refracted_background,color,white_opacity);
#endif
result=vec4(mix(water,color,coverage),1);}
'''
SURFACE_VELOCITY = SURFACE.replace('#version 330 core\n', '#version 330 core\n#define VELOCITY\n', 1)
