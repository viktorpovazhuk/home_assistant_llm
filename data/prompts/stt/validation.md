I used speech-to-text service to convert user voice command. Now I want to check the quality of recognition.
You are provided with a reference Sentence 1 and recognized Sentence 2. Check if:
1. They carry the same intent.
2. User asks to change the same properties of both devices. Set of these properties is the same in both sentences.
3. Device name consist of its location, type and number. Each component must be the same in both sentences. Lots of mistakes are made in device number, so pay attention to it.
Recognition system may use word "five" instead of number 5. Or reduction "it's" instead of "it is". Recognition system may loose words "please" or "now". These details doesn't matter. Pay attention to main points. But if device number is "4" and system recognized "for", it doesn't satisfy conditions.
Return "True" if sentences satisfy conditions. Otherwise, "False". Don't return anything else.