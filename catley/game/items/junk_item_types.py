"""Junk item type definitions.

All junk items are mechanically equivalent - usable for repair, sellable to merchants.
The variety is purely for environmental flavor (different items in different locations).
All items are plausibly mechanical/useful - no pure novelty items.
"""

import random

from catley.game.enums import ItemCategory, ItemSize
from catley.game.items.item_core import ItemType

# Raw data: (name, description)
_JUNK_DATA: list[tuple[str, str]] = [
    ("Assorted Screws", "A mixed jar of wood, metal, and machine screws."),
    ("Scrap Metal", "Jagged geometric shapes cut from larger sheets."),
    (
        "Copper Wire Spool",
        "Valuable for conductivity, though the insulation is stripping.",
    ),
    ("Duct Tape", "A half-used roll of silver adhesive savior."),
    ("Vacuum Tube", "Delicate glass component for retro-futuristic electronics."),
    ("Rusted Hinge", "Stiff, but salvageable steel."),
    ("Spark Plug", "Ceramic is stained, but the electrode is intact."),
    ("Frayed Power Cord", "The ends are exposed, good for splicing."),
    ("Ball Bearings", "A handful of greasy steel spheres."),
    ("Circuit Board", "Green silicone wafer with fried pathways."),
    ("Bicycle Chain", "Good for makeshift belts or melee weapon lashings."),
    ("Pressure Gauge", "The glass is cracked, but the needle still moves."),
    ("Solenoid", "An electromagnetic coil used for switches."),
    ("Compression Spring", "Bouncy metal coil from a vehicle suspension or mattress."),
    ("Hacksaw Blade", "Snapped in half, but the teeth are sharp."),
    ("Leather Strips", "Cut from an old jacket, essential for binding or padding."),
    ("Industrial Glue", "A tube of hardened, potent adhesive."),
    ("Fiber Optic Cable", "High-tech transmission line, severed."),
    ("Aluminum Heat Sink", "Finned metal block for cooling electronics."),
    ("Gear Cog", "A heavy brass gear with one missing tooth."),
    ("Electric Motor", "Small DC motor salvaged from a toy or appliance."),
    ("Rubber Hose", "Cracked tubing, useful for fluid transfer or siphoning."),
    ("Lens Assembly", "A stack of glass from a broken camera or scope."),
    ("Igniter Switch", "A push-button mechanism from a grill or explosive."),
    (
        "Grinding Wheel Fragment",
        "A chunk of abrasive stone snapped off a bench grinder.",
    ),
    ("Steel Wool", "Good for scrubbing rust or packing silencers."),
    ("Soldering Gun", "Broken tip, but the heating element might work."),
    ("Transistor Radio", "Gutted casing containing usable capacitors."),
    ("Fan Belt", "Rubber loop, stretched but intact."),
    ("Microchip", "A proprietary processor, potential goldmine for hackers."),
    ("O-Ring Set", "Rubber gaskets for sealing pipes or hydraulics."),
    ("Padlock", "Locked without a key, heavy enough to melt down or use as a weight."),
    ("Speaker Magnet", "Heavy donut-shaped magnet ripped from a woofer."),
    ("Fuse Box", "Contains a few unblown glass fuses."),
    ("Car Battery Terminal", "Lead connector crusted with acid."),
    ("Heating Element", "Coiled nichrome wire from a toaster or heater."),
    ("Zip Ties", "A bundle of brittle plastic fasteners."),
    ("Filing Rasp", "A metalworking tool with a broken handle."),
    ("9V Battery", "Corroded contacts, holds a ghost of a charge."),
    ("Toggle Switch", "A satisfying metal click-switch for machinery."),
    ("Piston", "Small engine part, oily and heavy."),
    ("Drill Bit Set", "Mostly dull, but high-carbon steel."),
    ("Epoxy Resin", "Two-part adhesive, slightly leaked."),
    ("Steel Pipe", "A foot-long section of threaded plumbing."),
    ("Canvas Patch", "Heavy fabric for repairing armor or backpacks."),
    ("Nylon Rope", "High tensile strength, frayed at the ends."),
    ("Tungsten Filament", "Extracted from a high-power lightbulb."),
    ("Motherboard Fragment", "Contains gold pins and salvageable solder."),
    ("Baling Wire", "Thin, pliable metal wire for quick fixes."),
    ("Graphics Card Fan", "Small plastic cooling fan."),
    ("Hose Clamps", "Metal rings with screw tighteners."),
    ("Broken Calipers", "Precision measuring tool, bent out of alignment."),
    ("Rifle Sling Swivel", "Metal attachment point for weapon straps."),
    ("Teflon Tape", "A spool of thin white tape used for sealing pipe threads."),
    ("Kevlar Scraps", "Fibrous material cut from a ruined vest."),
    ("Welding Rod", "A stick of filler metal for arc welding."),
    ("Laser Diode", "The red-eye component from a scanner or sight."),
    ("Servo Motor", "Precise robotic joint actuator."),
    ("Threaded Rod", "A long metal bolt without a head."),
    ("Steel Bracket", "L-shaped support metal."),
    ("Insulating Tape", "Black electrical tape, sticky residue on the sides."),
    ("Hard Drive Platter", "Mirror-finish disk, useless data but raw material."),
    ("Capacitor", "Cylindrical electronic component, risk of shock."),
    ("Resistor Pack", "Color-coded strips for regulating voltage."),
    ("Geiger Counter Probe", "The sensor wand, detached from the unit."),
    ("Bio-Scanner Screen", "Cracked LCD display."),
    ("Trigger Assembly", "The firing mechanism of a small firearm."),
    ("Firing Pin", "Machined steel pin, essential for ballistics."),
    ("Sandpaper", "A crumpled sheet of coarse grit."),
    ("Safety Glass Shard", "Tempered glass, useful for optics or blades."),
    ("Brass Elbow", "A heavy plumbing connector, good source of soft metal."),
    ("Mercury Switch", "Glass vial with liquid metal, for tilt triggers."),
    ("Propane Valve", "Brass fitting for gas regulation."),
    ("Alternator Coil", "Dense copper winding."),
    ("Ceramic Plate", "Shattered piece of body armor insert."),
    ("Rivets", "Industrial fasteners requiring a gun to install."),
    (
        "Lock Cylinder",
        "Extracted core of a door knob, containing tiny pins and springs.",
    ),
    ("Syringe Plunger", "Industrial grade, no needle."),
    ("Phone Battery", "Swollen lithium-ion pack."),
    ("Memory Stick", "Corrupted data, but salvageable contacts."),
    ("Solar Cell", "A single photovoltaic square, cracked."),
    ("LED Emitters", "Tiny light diodes, burnt out."),
    ("Coaxial Cable", "Shielded wire for signal transmission."),
    ("Thermostat Unit", "Old mechanical temperature control."),
    ("Chisel Tip", "The sharp end of a pneumatic tool."),
    ("Timing Belt", "Toothed rubber belt for engine calibration."),
    ("Valve Wheel", "Red iron handle for a spigot."),
    ("Wrench Jaw", "The adjustable head of a broken monkey wrench."),
    ("Sheet Metal Shears", "Dull blades, good pivot point."),
    ("Chain Link", "A single master link for chain repair."),
    ("Pull Cord", "Retractable starter cord for a generator."),
    ("Graphite Stick", "Lubricant or conductive material."),
    ("Magnetron", "The heavy microwave generator from an oven."),
    ("Fluorescent Starter", "Small canister to ignite gas lights."),
    ("Relay Switch", "Electromechanical switch block."),
    ("Utility Blade", "Razor sharp, rusted spine."),
    ("Dremel Bit", "A grinding stone tip for a rotary tool."),
    ("Watch Battery", "Tiny button cell, potentially holds charge."),
    ("Extension Cord Head", "Just the female socket end."),
    ("Antenna", "Telescoping metal rod from a radio."),
]

# Generate the ItemType instances
JUNK_ITEM_TYPES: list[ItemType] = [
    ItemType(
        name=name,
        description=desc,
        size=ItemSize.NORMAL,
        category=ItemCategory.JUNK,
    )
    for name, desc in _JUNK_DATA
]


def get_random_junk_type() -> ItemType:
    """Return a random junk item type for loot generation."""
    return random.choice(JUNK_ITEM_TYPES)
