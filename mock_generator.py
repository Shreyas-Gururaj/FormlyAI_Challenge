import os
import random

# Mock data templates for 510(k) summaries.
# This data is now richer to reflect the new section-based parsing.
MOCK_TEMPLATES = [
    {
        "k_number": "K230001",
        "device_name": "CardioScribe ECG Monitor",
        "manufacturer": "MedTech Solutions",
        "date_prepared": "01/15/2023",
        "contact_person": "Dr. Emily White",
        "classification_names": ["Electrocardiograph (21 CFR 870.2340)"],
        "common_name": "ECG Monitor",
        "proprietary_name": "CardioScribe v2",
        "predicate_devices": ["CardioTrack Model X (K210123)", "VitalView ECG (K200456)"],
        "device_description": "The CardioScribe ECG Monitor is a 12-lead portable diagnostic electrocardiograph. It consists of a patient acquisition module, a main processing unit with a 10-inch touchscreen display, and a built-in thermal printer. It is powered by a rechargeable lithium-ion battery or AC power. The device records, displays, and analyzes ECG waveforms and features digital signal processing to reduce artifact.",
        "intended_use": "The CardioScribe ECG Monitor is intended for use by qualified medical professionals to acquire, process, display, and record electrocardiographic (ECG) data from adult and pediatric patients.",
        "indications_for_use": "The CardioScribe ECG Monitor is indicated for use in measuring, recording, and analyzing ECG signals in adult and pediatric patients in a clinical setting. It is intended for use as a diagnostic tool to aid in the assessment of cardiac rhythm and morphology. It provides an interpretation of the ECG data, but all interpretations should be confirmed by a qualified physician.",
        "materials": "Patient-contacting components (electrodes) are made from standard Ag/AgCl with a hydrogel adhesive. The device housing is made from injection-molded ABS plastic.",
    },
    {
        "k_number": "K230002",
        "device_name": "OrthoFlex Hip System (Titanium)",
        "manufacturer": "JointRehab Inc.",
        "date_prepared": "02/20/2023",
        "contact_person": "Mr. David Chen",
        "classification_names": ["Hip joint metal/polymer/metal semi-constrained cemented or non-porous uncemented prosthesis (21 CFR 888.3353)"],
        "common_name": "Total Hip Replacement",
        "proprietary_name": "OrthoFlex Ti System",
        "predicate_devices": ["JointStep Hip Implant (K190234)", "Titan-Lock Hip (K200111)"],
        "device_description": "The OrthoFlex Hip System is a total hip arthroplasty (THA) system. It includes a femoral stem, a modular femoral head, an acetabular cup, and an acetabular liner. The femoral stem and acetabular cup are designed for cementless, press-fit fixation with a porous coating to promote bone ingrowth.",
        "intended_use": "The OrthoFlex Hip System is intended for use in total hip arthroplasty in skeletally mature patients to provide increased mobility and reduce pain.",
        "indications_for_use": "The OrthoFlex Hip System is indicated for patients with severe hip joint disease or damage due to causes such as non-inflammatory degenerative joint disease (e.g., osteoarthritis), rheumatoid arthritis, avascular necrosis, or fracture of the femoral neck. It is also indicated for the revision of previously failed hip arthroplasty.",
        "materials": "The femoral stem and modular head are made from wrought Titanium alloy (Ti-6Al-4V) conforming to ASTM F136. The acetabular cup is also Ti-6Al-4V. The liner is made from ultra-high-molecular-weight polyethylene (UHMWPE).",
    },
    {
        "k_number": "K230003",
        "device_name": "SonoView Fetal Ultrasound",
        "manufacturer": "Alpha Diagnostics",
        "date_prepared": "03/05/2023",
        "contact_person": "Dr. Sarah Jenkins",
        "classification_names": ["Diagnostic ultrasonic transducer (21 CFR 892.1570)", "Diagnostic ultrasound system (21 CFR 892.1550)"],
        "common_name": "Ultrasound System",
        "proprietary_name": "SonoView Fetal Ultrasound",
        "predicate_devices": ["EchoBaby 3000 (K200789)", "GynoScan Imager (K210333)"],
        "device_description": "The SonoView Fetal Ultrasound is a mobile, software-controlled diagnostic ultrasound system. It operates in B-Mode, M-Mode, and Pulsed Wave (PW) Doppler modes. The system includes a main console with a high-resolution 19-inch monitor, a keyboard, and multiple transducer ports. It is sold with a 3.5 MHz convex transducer (C3-5) and a 7.5 MHz transvaginal transducer (E7-5).",
        "intended_use": "The SonoView Fetal Ultrasound system is intended for use by trained healthcare professionals for diagnostic imaging in obstetrics, gynecology, and fetal imaging applications.",
        "indications_for_use": "The SonoView Fetal Ultrasound system is indicated for use in the visualization and assessment of the fetus, placenta, and maternal pelvic organs. This includes, but is not limited to, estimation of fetal age, assessment of fetal growth, detection of fetal anomalies, and assessment of fetal well-being.",
        "materials": "The transducer housing (patient-contacting) is made from medical-grade plastics (ABS and PC). The acoustic lens is made from a biocompatible silicone elastomer. All materials have been tested for biocompatibility per ISO 10993.",
    },
    {
        "k_number": "K230004",
        "device_name": "GlucoCheck Diabetes Monitor",
        "manufacturer": "Alpha Diagnostics",
        "date_prepared": "04/10/2023",
        "contact_person": "Dr. Sarah Jenkins",
        "classification_names": ["Glucose test system (21 CFR 862.1345)"],
        "common_name": "Blood Glucose Monitor",
        "proprietary_name": "GlucoCheck Now",
        "predicate_devices": ["Accu-Chek (K180001)", "OneTouch (K190002)"],
        "device_description": "The GlucoCheck Diabetes Monitor is a handheld, battery-powered in vitro diagnostic meter that works with GlucoCheck Test Strips to measure glucose (sugar) in whole blood. The system is intended for self-testing by people with diabetes.",
        "intended_use": "The GlucoCheck Diabetes Monitor is for the quantitative measurement of glucose in fresh capillary whole blood samples drawn from the fingertip, palm, or forearm.",
        "indications_for_use": "For use in the self-monitoring of blood glucose by individuals with diabetes mellitus. It is not intended for the diagnosis of or screening for diabetes, nor for use on neonates.",
        "materials": "Meter housing is PC/ABS plastic. Test strips are plastic with gold-plated electrodes and a glucose oxidase enzyme layer.",
    },
    {
        "k_number": "K230005",
        "device_name": "AcuSurg Scalpel",
        "manufacturer": "Surgical Innovations Ltd.",
        "date_prepared": "05/22/2023",
        "contact_person": "Ms. Priya Singh",
        "classification_names": ["Surgical scalpel (21 CFR 878.4800)"],
        "common_name": "Scalpel",
        "proprietary_name": "AcuSurg",
        "predicate_devices": ["Bard-Parker Safety Scalpel (K980001)"],
        "device_description": "The AcuSurg Scalpel is a sterile, single-use, disposable surgical scalpel. It features a manually retractable blade with a safety lock to prevent accidental sharps injuries. The handle is ergonomically designed and made of plastic. The blade is made of stainless steel.",
        "intended_use": "The AcuSurg Scalpel is intended for use in the operating room for cutting and dissecting tissue.",
        "indications_for_use": "For making incisions in tissue during surgical procedures. The safety feature is intended to aid in the prevention of sharps injuries.",
        "materials": "Blade is 1.4116 stainless steel. Handle is polystyrene.",
    },
    {
        "k_number": "K230006",
        "device_name": "DentaView X-Ray Sensor",
        "manufacturer": "Alpha Diagnostics",
        "date_prepared": "06/15/2023",
        "contact_person": "Dr. Sarah Jenkins",
        "classification_names": ["Intraoral X-ray sensor (21 CFR 872.1800)"],
        "common_name": "Dental X-Ray Sensor",
        "proprietary_name": "DentaView Sensor",
        "predicate_devices": ["Dexis Platinum Sensor (K170001)"],
        "device_description": "The DentaView X-Ray Sensor is a digital intraoral X-ray sensor. It is a CMOS-based sensor that connects via USB to a computer. It is designed to be used with standard dental X-ray generators. The sensor is housed in a hermetically sealed, water-resistant casing.",
        "intended_use": "The DentaView X-Ray Sensor is intended to be used by dentists and other qualified dental professionals to capture intraoral X-ray images.",
        "indications_for_use": "To produce digital images of dental structures to aid in the diagnosis of dental disease and to monitor oral health.",
        "materials": "Sensor housing is a biocompatible plastic (polyurethane) tested per ISO 10993. The cable is PVC-jacketed.",
    },
    {
        "k_number": "K063152",
        "device_name": "Surgi-OR Universal Surgical Kit",
        "manufacturer": "Surgical Innovations Ltd.",
        "date_prepared": "10/12/2006",
        "contact_person": "Johannes Mueller",
        "classification_names": ["Surgical tray (21 CFR 880.6850)"],
        "common_name": "Surgical Kit",
        "proprietary_name": "Surgi-OR Kit",
        "predicate_devices": ["K012345 Universal Pack"],
        "device_description": "The Surgi-OR Universal Surgical Kit is a sterile, single-use convenience kit containing common surgical instruments and supplies, such as scalpels, forceps, sutures, and gauze. It is intended to provide a standard set of tools for general surgical procedures.",
        "intended_use": "This kit is intended to be used by qualified healthcare professionals in a surgical setting as a convenience pack for general surgery.",
        "indications_for_use": "The Surgi-OR Universal Surgical Kit is indicated for use in general surgical procedures to provide the necessary single-use instruments and supplies. It is not intended for a specific surgical procedure.",
        "materials": "Instruments are stainless steel. Gauze is 100% cotton. Sutures are polyglycolic acid."
    }
]

OUTPUT_DIR = "./mock_510k_files"

def create_mock_files():
    """Generates mock 510(k) text files in the output directory."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Generating mock files in {OUTPUT_DIR}...")
    for item in MOCK_TEMPLATES:
        filename = f"{item['k_number']}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Create a more realistic-looking text file structure
        content = f"""
510(k) SUMMARY OF SAFETY AND EFFECTIVENESS

Document Number: {item['k_number']}
Date Prepared: {item['date_prepared']}

1. Submitter Information:
   {item['manufacturer']}
   123 Medical Device Rd.
   Innovation City, USA 12345

   Contact Person: {item['contact_person']}

2. Device Name:
   Trade Name: {item['proprietary_name']}
   Common Name: {item['common_name']}
   Classification Names: {'; '.join(item['classification_names'])}

3. Predicate Device(s):
   {', '.join(item['predicate_devices'])}

4. Device Description:
   {item['device_description']}

5. Intended Use:
   {item['intended_use']}

6. Indications for Use:
   {item['indications_for_use']}

7. Materials:
   {item['materials']}

8. Technological Characteristics:
   The technological characteristics of the {item['proprietary_name']} are substantially equivalent to the predicate device(s).
   Both use similar core principles of operation, materials, and energy sources.
   A comparison table highlights the minor differences which do not raise new questions of safety or effectiveness.

9. Performance Data:
   Bench testing and, if applicable, animal or clinical studies were conducted.
   Results demonstrate that the {item['proprietary_name']} meets all performance specifications and is as safe and effective as the predicate(s).
   Testing included electrical safety, biocompatibility, and software validation.

10. Conclusion:
    Based on the comparison of intended use, technological characteristics, and performance data, the {item['proprietary_name']} is substantially equivalent to the cited predicate devices.
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    
    # Add a few more simple files for variety
    simple_files = [
        ("K990001", "Simple Scalpel", "OldSurg Inc.", "A simple sterile scalpel.", "Cutting tissue.", "Cutting tissue.", "Stainless steel."),
        ("K990002", "Basic Syringe", "InjectCo", "A 10cc sterile syringe.", "Injecting fluids.", "Injecting or withdrawing fluids.", "Polypropylene plastic.")
    ]
    
    for k, name, mfg, desc, intended, indic, mat in simple_files:
        filename = f"{k}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        item = MOCK_TEMPLATES[0] # just to get some filler
        content = f"""
        510(k) SUMMARY
        K-NUMBER: {k}
        DEVICE NAME: {name}
        MANUFACTURER: {mfg}
        CONTACT: John Doe
        DATE: 01/01/1999
        PREDICATE(S): {item['predicate_devices'][0]}
        
        DEVICE DESCRIPTION: {desc}
        INTENDED USE: {intended}
        INDICATIONS FOR USE: {indic}
        MATERIALS: {mat}
        CONCLUSION: Substantially Equivalent.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Successfully generated {len(MOCK_TEMPLATES) + len(simple_files)} mock files.")

if __name__ == "__main__":
    create_mock_files()