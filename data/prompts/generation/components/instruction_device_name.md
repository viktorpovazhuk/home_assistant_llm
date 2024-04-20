You are provided with the device type and the user command. Firstly, remove any mentions of the place where device is located. Then, replace phrase that mentions device in the user command with a string {{DEVICE_NAME}}. Output must be strictly JSON object: {"Edited command 1": "...", "Edited command 2": "...", "Edited command 3": "..."}. Don't output anything else. There is an example below.

Device type 1: Humidity
User command 1: Update me on the humidity in the lounge.

Device type 2: Smoke
User command 2: Set the name of the smoke sensor to 'Office' now.

Device type 3: Temperature
User command 3: Set the temperature report threshold of the kitchen thermometer to 1C.

Device type 4: Input
User command 4: Tell me the status of the input in the guest room, please.

Device type 5: Light
User command 5: Disable auto on function for the office light.

Device type 6: Switch
User command 6: Check the instantaneous active power delivered by the switch in the bedroom.

{"Edited command 1": "Update me on the {{DEVICE_NAME}}.",
"Edited command 2": "Set the name of the {{DEVICE_NAME}} to 'Office' now.",
"Edited command 3": "Set the temperature report threshold of the {{DEVICE_NAME}} to 1C.",
"Edited command 4": "Tell me the status of the {{DEVICE_NAME}}, please.",
"Edited command 5": "Disable auto on function for the {{DEVICE_NAME}}.",
"Edited command 6": "Check the instantaneous active power delivered by the {{DEVICE_NAME}}."}