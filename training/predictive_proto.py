"""Wire-compatible Example with native/Python descriptor-pool API support."""
from google.protobuf import descriptor_pb2,descriptor_pool,message_factory

def example_type():
    """Wire-compatible Example descriptor; definitions verified against TF2.20."""
    schema=descriptor_pb2.FileDescriptorProto(name="slwm_probe_example.proto",package="slwm_probe",syntax="proto3")
    for name,kind in (("BytesList",12),("FloatList",2),("Int64List",3)):
        msg=schema.message_type.add(name=name); msg.field.add(name="value",number=1,label=3,type=kind)
    feature=schema.message_type.add(name="Feature"); feature.oneof_decl.add(name="kind")
    for i,name in enumerate(("BytesList","FloatList","Int64List"),1):
        field=feature.field.add(name=("bytes_list","float_list","int64_list")[i-1],number=i,label=1,type=11,type_name=".slwm_probe."+name); field.oneof_index=0
    features=schema.message_type.add(name="Features"); entry=features.nested_type.add(name="FeatureEntry"); entry.options.map_entry=True
    entry.field.add(name="key",number=1,label=1,type=9); entry.field.add(name="value",number=2,label=1,type=11,type_name=".slwm_probe.Feature")
    features.field.add(name="feature",number=1,label=3,type=11,type_name=".slwm_probe.Features.FeatureEntry")
    msg=schema.message_type.add(name="Example"); msg.field.add(name="features",number=1,label=1,type=11,type_name=".slwm_probe.Features")
    pool=descriptor_pool.DescriptorPool();pool.Add(schema)
    descriptor=pool.FindFileByName(schema.name)
    return message_factory.GetMessageClass(descriptor.message_types_by_name["Example"])

