medicinal_condition = " A medical condition refers to specific health issues, diseases, or physical states affected by plant-based or fungi-based treatments or compounds. Only label medical conditions that are treated using plants or fungi or associated health products or compounds."
medicinal_effect = " A medicinal effect refers to the therapeutic effects or properties attributed to plants and fungi, such as 'antitumor', 'anti-inflammatory', or 'digestive stimulant'. Only label medicinal effects that are produced by plants or fungi or associated health products or compounds."
formatting = ""
other_notes = "If you notice a bibliography or reference section in the text, you should ignore this."

prompt = ("From the chunk of text below, you should list all scientific plant and fungal names including authorship. "
          "For each of the plant or fungi names you should then list any medical conditions they treat or medicinal effects they have. "
          + medicinal_condition + medicinal_effect +
          " Your outputs should also be direct quotes from the text. "
          "As an example, in the following text:'Aloe vera L. is a very nice plant, it is used for lots of things. Though little evidence exists for it's efficacy, it is widely used to treat eczema as it is believed to reduce inflammation'. "
          "Your output should be: 1. Aloe vera L. Medical condition: eczema. Medicinal effect: reduce inflammation."
          ""
          + other_notes)
