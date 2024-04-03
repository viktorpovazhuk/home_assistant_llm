Retrieve response properties JSON object of the method below. Retrieve original JSON without any modifications. Don't hallucinate any properties. Don't omit any properties, even id property, because I will this JSON to validate my algorithm:

Method name: Temperature.SetConfig
Method description: Properties:
{"id": {"type": "number", "description": "Id of the Temperature component instance"}, "config": {"type": "object", "description": "Configuration that the method takes"}}
Find more about the config properties in config section
Temperature config object description:
The configuration of the Temperature component allows to adjust the temperature report threshold value. To Get/Set the configuration of the Temperature component its id must be specified.
Properties:
{"id": {"type": "number", "description": "Id of the Temperature component instance"}, "name": {"type": "string or null", "description": "Name of the Temperature instance. name length should not exceed 64 chars"}, "report_thr_C": {"type": "number", "description": "Temperature report threshold in Celsius. Accepted range is device-specific, default [0.5..5.0]C unless specified otherwise"}, "offset_C": {"type": "number", "description": "Offset in Celsius to be applied to the measured temperature. Accepted range is device-specific, default [-50.0 .. 50.0] unless specified otherwise"}}