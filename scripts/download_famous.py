"""
Download ~200 famous portrait paintings from Wikimedia Commons.

Enriches the existing dataset (data/portraits/) with iconic, recognizable
portrait paintings. Downloads via Wikimedia Commons API, resizes to 512x512,
appends metadata to portraits_metadata.csv with an is_famous column.
"""

import csv
import io
import logging
import os
import re
import time

import requests
from PIL import Image
from tqdm import tqdm

# Allow large images — many famous paintings on Commons are very high-res scans.
Image.MAX_IMAGE_PIXELS = 300_000_000

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "portraits")
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "portraits_metadata.csv")
TARGET_SIZE = (512, 512)
JPEG_QUALITY = 85

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FaceToArtBot/1.0 (portrait dataset enrichment; Python/requests)"
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 1.5  # be polite to Wikimedia servers
THUMB_WIDTH = 1024  # request thumbnails instead of full-res originals

# ---------------------------------------------------------------------------
# Hardcoded list of ~200 famous portrait paintings.
# Each entry: wikimedia_filename (exact Commons filename without "File:" prefix),
#             title, artist, style.
# ---------------------------------------------------------------------------

FAMOUS_PAINTINGS = [
    # ===== WESTERN CANON (~100) =====

    # Leonardo da Vinci
    {
        "wikimedia_filename": "Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg",
        "title": "Mona Lisa",
        "artist": "Leonardo da Vinci",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "The Lady with an Ermine.jpg",
        "title": "Lady with an Ermine",
        "artist": "Leonardo da Vinci",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "Leonardo da Vinci - Ginevra de' Benci - Google Art Project.jpg",
        "title": "Ginevra de' Benci",
        "artist": "Leonardo da Vinci",
        "style": "Renaissance",
    },

    # Vermeer
    {
        "wikimedia_filename": "1665 Girl with a Pearl Earring.jpg",
        "title": "Girl with a Pearl Earring",
        "artist": "Johannes Vermeer",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Johannes Vermeer - Het melkmeisje - Google Art Project.jpg",
        "title": "The Milkmaid",
        "artist": "Johannes Vermeer",
        "style": "Baroque",
    },

    # Rembrandt
    {
        "wikimedia_filename": "Rembrandt van Rijn - Self-Portrait - Google Art Project.jpg",
        "title": "Self-Portrait with Beret and Turned-Up Collar",
        "artist": "Rembrandt",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Rembrandt Harmensz. van Rijn 135.jpg",
        "title": "Self-Portrait with Two Circles",
        "artist": "Rembrandt",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Rembrandt - Self-Portrait with Velvet Beret - Google Art Project.jpg",
        "title": "Self-Portrait with Velvet Beret",
        "artist": "Rembrandt",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Rembrandt - Rembrandt and Saskia in the Scene of the Prodigal Son - Google Art Project.jpg",
        "title": "Self-Portrait as the Prodigal Son",
        "artist": "Rembrandt",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Rembrandt van Rijn 184.jpg",
        "title": "Self-Portrait 1659",
        "artist": "Rembrandt",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Rembrandt Harmensz. van Rijn 134.jpg",
        "title": "Self-Portrait at the Age of 63",
        "artist": "Rembrandt",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Rembrandt - The Anatomy Lesson of Dr Nicolaes Tulp.jpg",
        "title": "The Anatomy Lesson of Dr Nicolaes Tulp",
        "artist": "Rembrandt",
        "style": "Baroque",
    },

    # Van Gogh
    {
        "wikimedia_filename": "Vincent van Gogh - Self-Portrait - Google Art Project (454045).jpg",
        "title": "Self-Portrait",
        "artist": "Vincent van Gogh",
        "style": "Post-Impressionism",
    },
    {
        "wikimedia_filename": "Vincent Willem van Gogh 102.jpg",
        "title": "Self-Portrait with Bandaged Ear",
        "artist": "Vincent van Gogh",
        "style": "Post-Impressionism",
    },
    {
        "wikimedia_filename": "Vincent van Gogh - Self-portrait with grey felt hat - Google Art Project.jpg",
        "title": "Self-Portrait with Grey Felt Hat",
        "artist": "Vincent van Gogh",
        "style": "Post-Impressionism",
    },
    {
        "wikimedia_filename": "Vincent van Gogh - Dr Paul Gachet - Google Art Project.jpg",
        "title": "Portrait of Dr. Gachet",
        "artist": "Vincent van Gogh",
        "style": "Post-Impressionism",
    },
    {
        "wikimedia_filename": "Vincent van Gogh - Self-Portrait - Google Art Project.jpg",
        "title": "Self-Portrait (1889)",
        "artist": "Vincent van Gogh",
        "style": "Post-Impressionism",
    },

    # Frida Kahlo
    {
        "wikimedia_filename": "Frida Kahlo Self Portrait with Necklace of Thorns 1940.jpg",
        "title": "Self-Portrait with Thorn Necklace and Hummingbird",
        "artist": "Frida Kahlo",
        "style": "Surrealism",
    },
    {
        "wikimedia_filename": "Frida Kahlo Self-portrait with monkey 1938.jpg",
        "title": "Self-Portrait with Monkey",
        "artist": "Frida Kahlo",
        "style": "Surrealism",
    },
    {
        "wikimedia_filename": "Frida Kahlo - Self-portrait (1941) - Google Art Project.jpg",
        "title": "Self-Portrait (Frida Kahlo)",
        "artist": "Frida Kahlo",
        "style": "Surrealism",
    },
    {
        "wikimedia_filename": "The Two Fridas.jpg",
        "title": "The Two Fridas",
        "artist": "Frida Kahlo",
        "style": "Surrealism",
    },

    # Modigliani
    {
        "wikimedia_filename": "Amedeo-modigliani-jeanne-hebuterne-with-hat-and-necklace.jpg",
        "title": "Jeanne Hébuterne with Hat",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Amedeo Modigliani - Portrait de femme (1917-1918).jpg",
        "title": "Portrait de Femme",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Modigliani - Lunia Czechowska.jpg",
        "title": "Lunia Czechowska",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Amedeo Modigliani 017.jpg",
        "title": "Portrait of Leopold Zborowski",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Amedeo Modigliani 053.jpg",
        "title": "Nude Sitting on a Divan",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },

    # Edvard Munch
    {
        "wikimedia_filename": "Edvard Munch, 1893, The Scream, oil, tempera and pastel on cardboard, 91 x 73 cm, National Gallery of Norway.jpg",
        "title": "The Scream",
        "artist": "Edvard Munch",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Edvard Munch - Self-Portrait with Cigarette - NG.M.00470 - National Museum of Art, Architecture and Design.jpg",
        "title": "Self-Portrait with Cigarette",
        "artist": "Edvard Munch",
        "style": "Expressionism",
    },

    # Picasso
    {
        "wikimedia_filename": "Pablo Picasso, 1903-04, The Old Guitarist, oil on panel, 122.9 x 82.6 cm, Art Institute of Chicago.jpg",
        "title": "The Old Guitarist",
        "artist": "Pablo Picasso",
        "style": "Cubism",
    },
    {
        "wikimedia_filename": "Les Demoiselles d'Avignon (7925004644).jpg",
        "title": "Les Demoiselles d'Avignon",
        "artist": "Pablo Picasso",
        "style": "Cubism",
    },
    {
        "wikimedia_filename": "Pablo Picasso, 1901-02, Femme aux Bras Croisés, oil on canvas, 81 × 58 cm (32 × 23 in).jpg",
        "title": "Femme aux Bras Croisés",
        "artist": "Pablo Picasso",
        "style": "Cubism",
    },
    {
        "wikimedia_filename": "Pablo Picasso, 1905, Garçon à la pipe, (Boy with a Pipe), oil on canvas, 100 × 81.3 cm, private collection.jpg",
        "title": "Boy with a Pipe",
        "artist": "Pablo Picasso",
        "style": "Cubism",
    },
    {
        "wikimedia_filename": "Pablo Picasso, 1904, Paris, La Célestine (La Femme à la Taie), oil on canvas, 74.5 x 58.5 cm, Musée Picasso.jpg",
        "title": "La Celestine",
        "artist": "Pablo Picasso",
        "style": "Blue Period",
    },
    {
        "wikimedia_filename": "Pablo Picasso, 1903, The Tragedy, oil on wood, 105.3 cm × 69 cm (41.4 in × 27.2 in), National Gallery of Art, Washington, DC.jpg",
        "title": "The Tragedy",
        "artist": "Pablo Picasso",
        "style": "Blue Period",
    },

    # Gustav Klimt
    {
        "wikimedia_filename": "Gustav Klimt, 1907, Adele Bloch-Bauer I, Neue Galerie New York.jpg",
        "title": "Portrait of Adele Bloch-Bauer I",
        "artist": "Gustav Klimt",
        "style": "Art Nouveau",
    },
    {
        "wikimedia_filename": "The Kiss - Gustav Klimt - Google Cultural Institute.jpg",
        "title": "The Kiss",
        "artist": "Gustav Klimt",
        "style": "Art Nouveau",
    },
    {
        "wikimedia_filename": "Gustav Klimt 039.jpg",
        "title": "Judith and the Head of Holofernes",
        "artist": "Gustav Klimt",
        "style": "Art Nouveau",
    },

    # James McNeill Whistler
    {
        "wikimedia_filename": "Whistlers Mother high res.jpg",
        "title": "Whistler's Mother",
        "artist": "James McNeill Whistler",
        "style": "Realism",
    },

    # John Singer Sargent
    {
        "wikimedia_filename": "Madame X (Madame Pierre Gautreau), John Singer Sargent, 1884 (unfree frame crop).jpg",
        "title": "Madame X",
        "artist": "John Singer Sargent",
        "style": "Realism",
    },

    # Jan van Eyck
    {
        "wikimedia_filename": "Van Eyck - Arnolfini Portrait.jpg",
        "title": "Arnolfini Portrait",
        "artist": "Jan van Eyck",
        "style": "Northern Renaissance",
    },
    {
        "wikimedia_filename": "Portrait of a Man by Jan van Eyck-small.jpg",
        "title": "Portrait of a Man (Self-Portrait?)",
        "artist": "Jan van Eyck",
        "style": "Northern Renaissance",
    },

    # Albrecht Dürer
    {
        "wikimedia_filename": "Albrecht Dürer - Selbstbildnis im Pelzrock - Alte Pinakothek.jpg",
        "title": "Self-Portrait at Twenty-Eight",
        "artist": "Albrecht Dürer",
        "style": "Northern Renaissance",
    },
    {
        "wikimedia_filename": "Albrecht Dürer - 1498 Self-Portrait - Prado.jpg",
        "title": "Self-Portrait at Twenty-Six",
        "artist": "Albrecht Dürer",
        "style": "Northern Renaissance",
    },

    # Sandro Botticelli
    {
        "wikimedia_filename": "Sandro Botticelli - La nascita di Venere - Google Art Project - edited.jpg",
        "title": "The Birth of Venus",
        "artist": "Sandro Botticelli",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "Sandro Botticelli 080.jpg",
        "title": "Portrait of a Young Man (Botticelli)",
        "artist": "Sandro Botticelli",
        "style": "Renaissance",
    },

    # Raphael
    {
        "wikimedia_filename": "Raffaello Sanzio.jpg",
        "title": "Self-Portrait",
        "artist": "Raphael",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "Raphael - The Sistine Madonna - Google Art Project.jpg",
        "title": "Sistine Madonna",
        "artist": "Raphael",
        "style": "Renaissance",
    },

    # Caravaggio
    {
        "wikimedia_filename": "Narcissus-Caravaggio (1594-96) edited.jpg",
        "title": "Narcissus",
        "artist": "Caravaggio",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Caravaggio - Medusa - Google Art Project.jpg",
        "title": "Medusa",
        "artist": "Caravaggio",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Boy with a Basket of Fruit-Caravaggio (c.1593).jpg",
        "title": "Boy with a Basket of Fruit",
        "artist": "Caravaggio",
        "style": "Baroque",
    },

    # Diego Velázquez
    {
        "wikimedia_filename": "Las Meninas, by Diego Velázquez, from Prado in Google Earth.jpg",
        "title": "Las Meninas",
        "artist": "Diego Velázquez",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Retrato del Papa Inocencio X. Roma, by Diego Velázquez.jpg",
        "title": "Portrait of Innocent X",
        "artist": "Diego Velázquez",
        "style": "Baroque",
    },

    # René Magritte
    {
        "wikimedia_filename": "Magritte-Golconde.jpg",
        "title": "Golconda",
        "artist": "René Magritte",
        "style": "Surrealism",
    },

    # Salvador Dalí
    {
        "wikimedia_filename": "Salvador Dalí - Soft self-portrait with fried bacon.jpg",
        "title": "Soft Self-Portrait with Fried Bacon",
        "artist": "Salvador Dalí",
        "style": "Surrealism",
    },

    # Grant Wood
    {
        "wikimedia_filename": "Grant Wood - American Gothic - Google Art Project.jpg",
        "title": "American Gothic",
        "artist": "Grant Wood",
        "style": "Regionalism",
    },

    # Édouard Manet
    {
        "wikimedia_filename": "Edouard Manet - Olympia - Google Art Project 3.jpg",
        "title": "Olympia",
        "artist": "Édouard Manet",
        "style": "Impressionism",
    },
    {
        "wikimedia_filename": "Edouard Manet, A Bar at the Folies-Bergère.jpg",
        "title": "A Bar at the Folies-Bergère",
        "artist": "Édouard Manet",
        "style": "Impressionism",
    },

    # Pierre-Auguste Renoir
    {
        "wikimedia_filename": "Pierre-Auguste Renoir, Le Moulin de la Galette.jpg",
        "title": "Bal du moulin de la Galette",
        "artist": "Pierre-Auguste Renoir",
        "style": "Impressionism",
    },
    {
        "wikimedia_filename": "Pierre-Auguste Renoir - Luncheon of the Boating Party - Google Art Project.jpg",
        "title": "Luncheon of the Boating Party",
        "artist": "Pierre-Auguste Renoir",
        "style": "Impressionism",
    },

    # Edgar Degas
    {
        "wikimedia_filename": "Edgar Degas - In a Café - Google Art Project 2.jpg",
        "title": "L'Absinthe",
        "artist": "Edgar Degas",
        "style": "Impressionism",
    },

    # Paul Gauguin
    {
        "wikimedia_filename": "Paul Gauguin- Manao tupapau (The Spirit of the Dead Keep Watch).JPG",
        "title": "Spirit of the Dead Watching",
        "artist": "Paul Gauguin",
        "style": "Post-Impressionism",
    },
    {
        "wikimedia_filename": "Paul Gauguin 112.jpg",
        "title": "Self-Portrait with Halo and Snake",
        "artist": "Paul Gauguin",
        "style": "Post-Impressionism",
    },

    # Henri Matisse
    {
        "wikimedia_filename": "Matisse-Woman-with-a-Hat.jpg",
        "title": "Woman with a Hat",
        "artist": "Henri Matisse",
        "style": "Fauvism",
    },
    {
        "wikimedia_filename": "Henri Matisse, 1906, Self-Portrait in a Striped T-shirt, oil on canvas, 55 x 46 cm, Statens Museum for Kunst, Copenhagen.jpg",
        "title": "Self-Portrait in a Striped T-shirt",
        "artist": "Henri Matisse",
        "style": "Fauvism",
    },

    # Egon Schiele
    {
        "wikimedia_filename": "Egon Schiele - Self-Portrait with Physalis - Google Art Project.jpg",
        "title": "Self-Portrait with Physalis",
        "artist": "Egon Schiele",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Egon Schiele 079.jpg",
        "title": "Portrait of Wally Neuzil",
        "artist": "Egon Schiele",
        "style": "Expressionism",
    },

    # Hans Holbein
    {
        "wikimedia_filename": "Hans Holbein the Younger - The Ambassadors - Google Art Project.jpg",
        "title": "The Ambassadors",
        "artist": "Hans Holbein the Younger",
        "style": "Northern Renaissance",
    },
    {
        "wikimedia_filename": "Hans Holbein d. J. 065.jpg",
        "title": "Portrait of Henry VIII",
        "artist": "Hans Holbein the Younger",
        "style": "Northern Renaissance",
    },

    # Titian
    {
        "wikimedia_filename": "Tizian 090.jpg",
        "title": "Venus of Urbino",
        "artist": "Titian",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "Titian - Portrait of a man with a quilted sleeve.jpg",
        "title": "Portrait of a Man with a Quilted Sleeve",
        "artist": "Titian",
        "style": "Renaissance",
    },

    # El Greco
    {
        "wikimedia_filename": "El caballero de la mano en el pecho, by El Greco, from Prado in Google Earth.jpg",
        "title": "The Nobleman with his Hand on his Chest",
        "artist": "El Greco",
        "style": "Mannerism",
    },

    # Francisco Goya
    {
        "wikimedia_filename": "Francisco de Goya, Pair of Majas on a Balcony.jpg",
        "title": "Majas on a Balcony",
        "artist": "Francisco Goya",
        "style": "Romanticism",
    },
    {
        "wikimedia_filename": "Duchess of Alba or The White Duchess by Goya.jpg",
        "title": "Portrait of the Duchess of Alba",
        "artist": "Francisco Goya",
        "style": "Romanticism",
    },

    # Eugène Delacroix
    {
        "wikimedia_filename": "Eugène Delacroix - La liberté guidant le peuple.jpg",
        "title": "Liberty Leading the People",
        "artist": "Eugène Delacroix",
        "style": "Romanticism",
    },

    # Jacques-Louis David
    {
        "wikimedia_filename": "Death of Marat by David.jpg",
        "title": "The Death of Marat",
        "artist": "Jacques-Louis David",
        "style": "Neoclassicism",
    },
    {
        "wikimedia_filename": "Jacques-Louis David - The Emperor Napoleon in His Study at the Tuileries - Google Art Project.jpg",
        "title": "The Emperor Napoleon in His Study",
        "artist": "Jacques-Louis David",
        "style": "Neoclassicism",
    },

    # Jean-Auguste-Dominique Ingres
    {
        "wikimedia_filename": "Jean Auguste Dominique Ingres, La Grande Odalisque, 1814.jpg",
        "title": "La Grande Odalisque",
        "artist": "Jean-Auguste-Dominique Ingres",
        "style": "Neoclassicism",
    },

    # Artemisia Gentileschi
    {
        "wikimedia_filename": "Self-portrait as the Allegory of Painting (La Pittura) - Artemisia Gentileschi.jpg",
        "title": "Self-Portrait as the Allegory of Painting",
        "artist": "Artemisia Gentileschi",
        "style": "Baroque",
    },

    # Elisabeth Vigée Le Brun
    {
        "wikimedia_filename": "Self-portrait in a Straw Hat by Elisabeth-Louise Vigée-Lebrun.jpg",
        "title": "Self-Portrait in a Straw Hat",
        "artist": "Elisabeth Vigée Le Brun",
        "style": "Neoclassicism",
    },
    {
        "wikimedia_filename": "Louise Elisabeth Vigée-Lebrun - Marie-Antoinette dit « à la Rose » - Google Art Project.jpg",
        "title": "Marie Antoinette with a Rose",
        "artist": "Elisabeth Vigée Le Brun",
        "style": "Neoclassicism",
    },

    # Parmigianino
    {
        "wikimedia_filename": "Parmigianino Selfportrait.jpg",
        "title": "Self-Portrait in a Convex Mirror",
        "artist": "Parmigianino",
        "style": "Mannerism",
    },

    # Jean-Honoré Fragonard
    {
        "wikimedia_filename": "Fragonard, The Reader.jpg",
        "title": "The Reader",
        "artist": "Jean-Honoré Fragonard",
        "style": "Rococo",
    },

    # Thomas Gainsborough
    {
        "wikimedia_filename": "Thomas Gainsborough - The Blue Boy (The Huntington Library, San Marino L. A.).jpg",
        "title": "The Blue Boy",
        "artist": "Thomas Gainsborough",
        "style": "Rococo",
    },

    # John Everett Millais
    {
        "wikimedia_filename": "John Everett Millais - Ophelia - Google Art Project.jpg",
        "title": "Ophelia",
        "artist": "John Everett Millais",
        "style": "Pre-Raphaelite",
    },

    # Dante Gabriel Rossetti
    {
        "wikimedia_filename": "Dante Gabriel Rossetti - Proserpine - Google Art Project.jpg",
        "title": "Proserpine",
        "artist": "Dante Gabriel Rossetti",
        "style": "Pre-Raphaelite",
    },

    # Amedeo Modigliani (more)
    {
        "wikimedia_filename": "Amedeo Modigliani - Portrait of the Painter Chaim Soutine.jpg",
        "title": "Portrait of Chaim Soutine",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },

    # Henri de Toulouse-Lautrec
    {
        "wikimedia_filename": "Henri de Toulouse-Lautrec 031.jpg",
        "title": "At the Moulin Rouge",
        "artist": "Henri de Toulouse-Lautrec",
        "style": "Post-Impressionism",
    },

    # Paul Cézanne
    {
        "wikimedia_filename": "Paul Cézanne 159.jpg",
        "title": "Self-Portrait with Beret",
        "artist": "Paul Cézanne",
        "style": "Post-Impressionism",
    },

    # Edouard Manet
    {
        "wikimedia_filename": "Édouard Manet - Le Déjeuner sur l'herbe.jpg",
        "title": "Le Déjeuner sur l'herbe",
        "artist": "Édouard Manet",
        "style": "Impressionism",
    },

    # Mary Cassatt
    {
        "wikimedia_filename": "Mary Cassatt - The Child's Bath - Google Art Project.jpg",
        "title": "The Child's Bath",
        "artist": "Mary Cassatt",
        "style": "Impressionism",
    },

    # Berthe Morisot
    {
        "wikimedia_filename": "Berthe Morisot - The Cradle - Google Art Project.jpg",
        "title": "The Cradle",
        "artist": "Berthe Morisot",
        "style": "Impressionism",
    },

    # Andy Warhol
    {
        "wikimedia_filename": "Andy Warhol by Jack Mitchell.jpg",
        "title": "Andy Warhol Portrait",
        "artist": "Andy Warhol",
        "style": "Pop Art",
    },

    # Francis Bacon
    {
        "wikimedia_filename": "Self-portrait, Pablo Picasso, 1907, Národní galerie v Praze.jpg",
        "title": "Self-Portrait (1907)",
        "artist": "Pablo Picasso",
        "style": "Cubism",
    },

    # Lucian Freud
    {
        "wikimedia_filename": "Chaim Soutine - Carcass of Beef - Google Art Project.jpg",
        "title": "Carcass of Beef",
        "artist": "Chaim Soutine",
        "style": "Expressionism",
    },

    # Edward Hopper
    {
        "wikimedia_filename": "Nighthawks by Edward Hopper 1942.jpg",
        "title": "Nighthawks",
        "artist": "Edward Hopper",
        "style": "Realism",
    },

    # Andrew Wyeth
    {
        "wikimedia_filename": "Winslow Homer - Breezing Up (A Fair Wind) - Google Art Project.jpg",
        "title": "Breezing Up (A Fair Wind)",
        "artist": "Winslow Homer",
        "style": "Realism",
    },

    # Tamara de Lempicka
    {
        "wikimedia_filename": "Tamara de Lempicka - Kizette en rose, 1926.jpg",
        "title": "Kizette en Rose",
        "artist": "Tamara de Lempicka",
        "style": "Art Deco",
    },

    # Peter Paul Rubens
    {
        "wikimedia_filename": "Le Chapeau de Paille by Peter Paul Rubens.jpg",
        "title": "Le Chapeau de Paille",
        "artist": "Peter Paul Rubens",
        "style": "Baroque",
    },

    # Anthony van Dyck
    {
        "wikimedia_filename": "Anthony van Dyck - Charles I at the Hunt (cropped).jpg",
        "title": "Charles I at the Hunt",
        "artist": "Anthony van Dyck",
        "style": "Baroque",
    },

    # Rogier van der Weyden
    {
        "wikimedia_filename": "Rogier van der Weyden - Portrait of a Lady - Google Art Project.jpg",
        "title": "Portrait of a Lady",
        "artist": "Rogier van der Weyden",
        "style": "Northern Renaissance",
    },

    # Petrus Christus
    {
        "wikimedia_filename": "Petrus Christus - Portrait of a Young Woman - Google Art Project.jpg",
        "title": "Portrait of a Young Girl",
        "artist": "Petrus Christus",
        "style": "Northern Renaissance",
    },

    # Gustave Courbet
    {
        "wikimedia_filename": "Gustave Courbet - Le Désespéré (1843).jpg",
        "title": "The Desperate Man",
        "artist": "Gustave Courbet",
        "style": "Realism",
    },

    # Giorgione
    {
        "wikimedia_filename": "Giorgione - La Vecchia.jpg",
        "title": "La Vecchia",
        "artist": "Giorgione",
        "style": "Renaissance",
    },

    # Antonello da Messina
    {
        "wikimedia_filename": "Antonello da Messina - Portrait of a Man - National Gallery London.jpg",
        "title": "Portrait of a Man",
        "artist": "Antonello da Messina",
        "style": "Renaissance",
    },

    # Giuseppe Arcimboldo
    {
        "wikimedia_filename": "Vertumnus årstidernas gud målad av Giuseppe Arcimboldo 1591 - Skoklosters slott - 91503.jpg",
        "title": "Vertumnus",
        "artist": "Giuseppe Arcimboldo",
        "style": "Mannerism",
    },

    # ===== RUSSIAN (~25) =====

    # Ivan Kramskoy
    {
        "wikimedia_filename": "Kramskoy Unknown woman.jpg",
        "title": "Portrait of an Unknown Woman",
        "artist": "Ivan Kramskoy",
        "style": "Realism",
    },
    {
        "wikimedia_filename": "Kramskoy Christ dans le désert.jpg",
        "title": "Christ in the Desert",
        "artist": "Ivan Kramskoy",
        "style": "Realism",
    },

    # Valentin Serov
    {
        "wikimedia_filename": "Valentin Serov - Девочка с персиками. Портрет В.С.Мамонтовой - Google Art Project.jpg",
        "title": "Girl with Peaches",
        "artist": "Valentin Serov",
        "style": "Impressionism",
    },
    {
        "wikimedia_filename": "Valentin Serov - Ida Rubenstein - Google Art Project.jpg",
        "title": "Portrait of Ida Rubinstein",
        "artist": "Valentin Serov",
        "style": "Art Nouveau",
    },
    {
        "wikimedia_filename": "Walentin Alexandrowitsch Serow 003.jpg",
        "title": "Girl Illuminated by the Sun",
        "artist": "Valentin Serov",
        "style": "Impressionism",
    },

    # Mikhail Vrubel
    {
        "wikimedia_filename": "Vrubel Demon.jpg",
        "title": "Demon Seated",
        "artist": "Mikhail Vrubel",
        "style": "Symbolism",
    },
    {
        "wikimedia_filename": "Tsarevna-Lebed by Mikhail Vrubel (brightened).jpg",
        "title": "The Swan Princess",
        "artist": "Mikhail Vrubel",
        "style": "Symbolism",
    },

    # Orest Kiprensky
    {
        "wikimedia_filename": "Kiprensky Pushkin.jpg",
        "title": "Portrait of Alexander Pushkin",
        "artist": "Orest Kiprensky",
        "style": "Romanticism",
    },

    # Ilya Repin
    {
        "wikimedia_filename": "REPIN Ivan Terrible&Ivan.jpg",
        "title": "Ivan the Terrible and His Son Ivan",
        "artist": "Ilya Repin",
        "style": "Realism",
    },
    {
        "wikimedia_filename": "Moscou, galerie Tretiakov Mussorgsky by Ilya Repin.jpg",
        "title": "Portrait of Modest Mussorgsky",
        "artist": "Ilya Repin",
        "style": "Realism",
    },
    {
        "wikimedia_filename": "Ilia Efimovich Repin (1844-1930) - Volga Boatmen (1870-1873).jpg",
        "title": "Barge Haulers on the Volga",
        "artist": "Ilya Repin",
        "style": "Realism",
    },
    {
        "wikimedia_filename": "Repin Cossacks.jpg",
        "title": "Reply of the Zaporozhian Cossacks",
        "artist": "Ilya Repin",
        "style": "Realism",
    },

    # Konstantin Flavitsky
    {
        "wikimedia_filename": "Tarakanova.jpg",
        "title": "Princess Tarakanova",
        "artist": "Konstantin Flavitsky",
        "style": "Romanticism",
    },

    # Kazimir Malevich
    {
        "wikimedia_filename": "Kazimir Malevich, 1933, Self Portrait, oil on canvas, 73 x 66 cm, State Russian Museum.jpg",
        "title": "Self-Portrait",
        "artist": "Kazimir Malevich",
        "style": "Suprematism",
    },

    # Vasily Surikov
    {
        "wikimedia_filename": "Surikov streltsi.jpg",
        "title": "Morning of the Execution of the Streltsy",
        "artist": "Vasily Surikov",
        "style": "Realism",
    },
    {
        "wikimedia_filename": "Vasily Surikov - Боярыня Морозова - Google Art Project.jpg",
        "title": "Boyarina Morozova",
        "artist": "Vasily Surikov",
        "style": "Realism",
    },

    # Karl Bryullov
    {
        "wikimedia_filename": "1832. BRULLOV VSADNICA1.jpg",
        "title": "The Horsewoman",
        "artist": "Karl Bryullov",
        "style": "Romanticism",
    },
    {
        "wikimedia_filename": "Karl Brullov, Italian midday, 1827.jpg",
        "title": "Italian Midday",
        "artist": "Karl Bryullov",
        "style": "Romanticism",
    },

    # Viktor Vasnetsov
    {
        "wikimedia_filename": "Viktor Vasnetsov - Аленушка - Google Art Project.jpg",
        "title": "Alyonushka",
        "artist": "Viktor Vasnetsov",
        "style": "Romanticism",
    },

    # Vasily Tropinin
    {
        "wikimedia_filename": "Vasily Tropinin - Кружевница - Google Art Project.jpg",
        "title": "The Lace Maker",
        "artist": "Vasily Tropinin",
        "style": "Romanticism",
    },

    # Ivan Argunov
    {
        "wikimedia_filename": "Ivan Argunov - Портрет неизвестной в русском костюме - Google Art Project.jpg",
        "title": "Portrait of an Unknown Woman in Russian Costume",
        "artist": "Ivan Argunov",
        "style": "Rococo",
    },

    # Boris Kustodiev
    {
        "wikimedia_filename": "Boris Kustodiev - Merchant's Wife at Tea - Google Art Project.jpg",
        "title": "Merchant's Wife at Tea",
        "artist": "Boris Kustodiev",
        "style": "Art Nouveau",
    },

    # Zinaida Serebriakova
    {
        "wikimedia_filename": "Sinaida Jewgenjewna Serebrjakowa At the Dressing-Table 1909.jpg",
        "title": "At the Dressing Table (Self-Portrait)",
        "artist": "Zinaida Serebriakova",
        "style": "Impressionism",
    },

    # Dmitry Levitzky
    {
        "wikimedia_filename": "Dmitry Levitsky - Портрет графини Урсулы Мнишек - Google Art Project.jpg",
        "title": "Portrait of Ursula Mniszech",
        "artist": "Dmitry Levitzky",
        "style": "Neoclassicism",
    },

    # Vladimir Borovikovsky
    {
        "wikimedia_filename": "Borovikovsky maria Lopukhina.jpg",
        "title": "Portrait of Maria Lopukhina",
        "artist": "Vladimir Borovikovsky",
        "style": "Neoclassicism",
    },

    # ===== JAPANESE / EAST ASIAN (~25) =====

    # Kitagawa Utamaro
    {
        "wikimedia_filename": "Utamaro2.jpg",
        "title": "Three Beauties of the Present Day",
        "artist": "Kitagawa Utamaro",
        "style": "Ukiyo-e",
    },
    {
        "wikimedia_filename": "Kitagawa Utamaro - Two Beauties - 14.76.70b - Metropolitan Museum of Art.jpg",
        "title": "Beauty Looking Back",
        "artist": "Kitagawa Utamaro",
        "style": "Ukiyo-e",
    },
    {
        "wikimedia_filename": "Bijin-ga (美人画), by Kitagawa Utamaro (喜多川歌麿).jpg",
        "title": "Large Head Beauty",
        "artist": "Kitagawa Utamaro",
        "style": "Ukiyo-e",
    },
    {
        "wikimedia_filename": "Kitagawa Utamaro (喜多川 歌麿) - Woman Reading Letter - Google Art Project.jpg",
        "title": "A Beauty",
        "artist": "Kitagawa Utamaro",
        "style": "Ukiyo-e",
    },
    {
        "wikimedia_filename": "Kitagawa Utamaro 001.jpg",
        "title": "Woman Wiping Sweat",
        "artist": "Kitagawa Utamaro",
        "style": "Ukiyo-e",
    },

    # Toshusai Sharaku
    {
        "wikimedia_filename": "Toshusai Sharaku- Otani Oniji, 1794.jpg",
        "title": "Otani Oniji III as Yakko Edobei",
        "artist": "Toshusai Sharaku",
        "style": "Ukiyo-e",
    },
    {
        "wikimedia_filename": "Sharaku2.jpg",
        "title": "Ichikawa Ebizo as Takemura Sadanoshin",
        "artist": "Toshusai Sharaku",
        "style": "Ukiyo-e",
    },
    {
        "wikimedia_filename": "Sharaku1.jpg",
        "title": "Segawa Tomisaburo II as Yadorigi",
        "artist": "Toshusai Sharaku",
        "style": "Ukiyo-e",
    },

    # Katsushika Hokusai
    {
        "wikimedia_filename": "Hokusai 1760-1849 Ocean waves.jpg",
        "title": "The Great Wave off Kanagawa",
        "artist": "Katsushika Hokusai",
        "style": "Ukiyo-e",
    },

    # Utagawa Kuniyoshi
    {
        "wikimedia_filename": "Kuniyoshi Utagawa, The fifty three stations of the tokaido.jpg",
        "title": "Station Nihonbashi",
        "artist": "Utagawa Kuniyoshi",
        "style": "Ukiyo-e",
    },

    # Suzuki Harunobu
    {
        "wikimedia_filename": "Suzuki Harunobu - Woman Visiting the Shrine in the Night - Google Art Project.jpg",
        "title": "Woman Visiting the Shrine in the Night",
        "artist": "Suzuki Harunobu",
        "style": "Ukiyo-e",
    },

    # Tsukioka Yoshitoshi
    {
        "wikimedia_filename": "Tsukioka Yoshitoshi - 100 Aspects of the Moon - 1.jpg",
        "title": "Looking in Pain",
        "artist": "Tsukioka Yoshitoshi",
        "style": "Ukiyo-e",
    },

    # Torii Kiyonaga
    {
        "wikimedia_filename": "Torii Kiyonaga - Two Women at the Veranda - Google Art Project.jpg",
        "title": "Beauties in the Rain",
        "artist": "Torii Kiyonaga",
        "style": "Ukiyo-e",
    },

    # Keisai Eisen
    {
        "wikimedia_filename": "Eisen1.jpg",
        "title": "Bijin-ga",
        "artist": "Keisai Eisen",
        "style": "Ukiyo-e",
    },

    # Chinese / Tang Dynasty
    {
        "wikimedia_filename": "Court Ladies Preparing Newly Woven Silk.jpg",
        "title": "Court Ladies Preparing Newly Woven Silk",
        "artist": "Zhang Xuan (attr.)",
        "style": "Tang Dynasty",
    },
    {
        "wikimedia_filename": "Gu Kaizhi 001.jpg",
        "title": "Admonitions of the Instructress to the Court Ladies (detail)",
        "artist": "Gu Kaizhi (attr.)",
        "style": "Chinese Classical",
    },

    # Joseon Dynasty Korea
    {
        "wikimedia_filename": "Yi Jaegwan - Portrait of Scholar-Official An in his Fiftieth Year - Google Art Project.jpg",
        "title": "Portrait of a Scholar-Official",
        "artist": "Anonymous (Joseon)",
        "style": "Joseon Dynasty",
    },
    {
        "wikimedia_filename": "Hyewon-Ssanggeum.daemu.jpg",
        "title": "Ssanggeum Daemu (Sword Dance)",
        "artist": "Shin Yun-bok",
        "style": "Joseon Dynasty",
    },
    {
        "wikimedia_filename": "Hyewon-Miindo.jpg",
        "title": "Portrait of a Beauty",
        "artist": "Shin Yun-bok",
        "style": "Joseon Dynasty",
    },

    # Utagawa Hiroshige
    {
        "wikimedia_filename": "Hiroshige, Evening snow at Kambara, ca. 1833-1834.jpg",
        "title": "Evening Snow at Kanbara",
        "artist": "Utagawa Hiroshige",
        "style": "Ukiyo-e",
    },

    # Itō Jakuchū
    {
        "wikimedia_filename": "Itō Jakuchū - Rooster and Hen with Hydrangeas (Colorful Realm of Living Beings).jpg",
        "title": "Rooster and Hen with Hydrangeas",
        "artist": "Itō Jakuchū",
        "style": "Edo Period",
    },

    # ===== OTHER CULTURES (~25) =====

    # Fayum Mummy Portraits (Roman Egypt)
    {
        "wikimedia_filename": "Fayum-01.jpg",
        "title": "Fayum Mummy Portrait (Young Man)",
        "artist": "Anonymous (Roman Egypt)",
        "style": "Fayum Portrait",
    },
    {
        "wikimedia_filename": "Fayum-34.jpg",
        "title": "Fayum Mummy Portrait of a Woman",
        "artist": "Anonymous (Roman Egypt)",
        "style": "Fayum Portrait",
    },
    {
        "wikimedia_filename": "Fayum-35.jpg",
        "title": "Fayum Mummy Portrait (Bearded Man)",
        "artist": "Anonymous (Roman Egypt)",
        "style": "Fayum Portrait",
    },
    {
        "wikimedia_filename": "EncausticPortraitWoman.jpg",
        "title": "Encaustic Portrait of a Woman",
        "artist": "Anonymous (Roman Egypt)",
        "style": "Fayum Portrait",
    },
    {
        "wikimedia_filename": "Portrait of a young man of the Fayum type, Inv. 56605 (Vatican Museums).jpg",
        "title": "Portrait of a Young Man (Fayum)",
        "artist": "Anonymous (Roman Egypt)",
        "style": "Fayum Portrait",
    },

    # Mughal Miniatures
    {
        "wikimedia_filename": "Jahangir investing a courtier with a robe of honour watched by Sir Thomas Roe, English ambassador to the court of Jahangir at Agra from 1615-18, and others.jpg",
        "title": "Jahangir Investing a Courtier",
        "artist": "Abu'l Hasan (attr.)",
        "style": "Mughal Miniature",
    },
    {
        "wikimedia_filename": "Portrait of Emperor Akbar Praying.jpg",
        "title": "Portrait of Emperor Akbar",
        "artist": "Anonymous (Mughal)",
        "style": "Mughal Miniature",
    },
    {
        "wikimedia_filename": "Shah Jahan on The Peacock Throne.jpg",
        "title": "Shah Jahan on the Peacock Throne",
        "artist": "Anonymous (Mughal)",
        "style": "Mughal Miniature",
    },
    {
        "wikimedia_filename": "Portrait of Mumtaz Mahal on Ivory.jpg",
        "title": "Portrait of Mumtaz Mahal",
        "artist": "Anonymous (Mughal)",
        "style": "Mughal Miniature",
    },

    # Ethiopian Icons
    {
        "wikimedia_filename": "Ethiopia Madonna and Child.jpg",
        "title": "Ethiopian Madonna and Child",
        "artist": "Anonymous (Ethiopian)",
        "style": "Ethiopian Icon",
    },
    {
        "wikimedia_filename": "Ethiopian - Triptych Center Panel with Mary and Her Son and Christ Teaching the Apostles - Walters 367 - Open.jpg",
        "title": "Ethiopian Triptych Panel",
        "artist": "Anonymous (Ethiopian)",
        "style": "Ethiopian Icon",
    },

    # Mexican Muralism
    {
        "wikimedia_filename": "Diego Rivera - The Flower Vendor - Google Art Project.jpg",
        "title": "The Flower Vendor",
        "artist": "Diego Rivera",
        "style": "Mexican Muralism",
    },
    {
        "wikimedia_filename": "'El Coronelazo (self-portrait)' (1945) by David Alfaro Siqueiros - Museo Nacional de Artes - Mexico 2024.jpg",
        "title": "El Coronelazo (Self-Portrait)",
        "artist": "David Alfaro Siqueiros",
        "style": "Mexican Muralism",
    },

    # Byzantine
    {
        "wikimedia_filename": "Vladimirskaya.jpg",
        "title": "Theotokos of Vladimir",
        "artist": "Anonymous (Byzantine)",
        "style": "Byzantine Icon",
    },
    {
        "wikimedia_filename": "Christ Pantocrator Deesis mosaic Hagia Sophia.jpg",
        "title": "Christ Pantocrator (Hagia Sophia)",
        "artist": "Anonymous (Byzantine)",
        "style": "Byzantine Mosaic",
    },
    {
        "wikimedia_filename": "Angelos Akotantos - The Virgin Cardiotissa - WGA00097.jpg",
        "title": "The Virgin Cardiotissa",
        "artist": "Angelos Akotantos",
        "style": "Byzantine Icon",
    },

    # Persian Miniatures
    {
        "wikimedia_filename": "Reza Abbasi - Two Lovers (1630).jpg",
        "title": "Two Lovers",
        "artist": "Reza Abbasi",
        "style": "Persian Miniature",
    },
    {
        "wikimedia_filename": "Suleiman the Magnificent.jpg",
        "title": "Suleiman the Magnificent",
        "artist": "Anonymous (Ottoman)",
        "style": "Ottoman Miniature",
    },

    # African
    {
        "wikimedia_filename": "Irma Stern - Arab with Oranges - Google Art Project.jpg",
        "title": "Arab with Oranges",
        "artist": "Irma Stern",
        "style": "Expressionism",
    },

    # Oceania
    {
        "wikimedia_filename": "Paul Gauguin - D'ou venons-nous.jpg",
        "title": "Where Do We Come From? What Are We? Where Are We Going?",
        "artist": "Paul Gauguin",
        "style": "Post-Impressionism",
    },

    # Australian
    {
        "wikimedia_filename": "Tom Roberts - Shearing the rams - Google Art Project.jpg",
        "title": "Shearing the Rams",
        "artist": "Tom Roberts",
        "style": "Impressionism",
    },

    # Latin America
    {
        "wikimedia_filename": "Rufino Tamayo - Found Objects Number 1 - Google Art Project.jpg",
        "title": "Found Objects Number 1",
        "artist": "Rufino Tamayo",
        "style": "Modernism",
    },

    # Additional Western to fill gaps
    {
        "wikimedia_filename": "Jan Vermeer - The Art of Painting - Google Art Project.jpg",
        "title": "The Art of Painting",
        "artist": "Johannes Vermeer",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Michelangelo's Pieta 5450 cropncleaned edit.jpg",
        "title": "Pieta",
        "artist": "Michelangelo",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "Michelangelo - Creation of Adam (cropped).jpg",
        "title": "Creation of Adam",
        "artist": "Michelangelo",
        "style": "Renaissance",
    },
    {
        "wikimedia_filename": "Giotto - Scrovegni - -31- - Kiss of Judas.jpg",
        "title": "Kiss of Judas",
        "artist": "Giotto di Bondone",
        "style": "Proto-Renaissance",
    },
    {
        "wikimedia_filename": "Gerrit van Honthorst - Smiling Girl, a Courtesan, Holding an Obscene Image - 63-1954 - Saint Louis Art Museum.jpg",
        "title": "Smiling Girl, a Courtesan",
        "artist": "Gerard van Honthorst",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Judith Leyster - A Boy and a Girl with a Cat and an Eel - WGA12956.jpg",
        "title": "A Boy and a Girl with a Cat and an Eel",
        "artist": "Judith Leyster",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Cavalier soldier Hals-1624x.jpg",
        "title": "The Laughing Cavalier",
        "artist": "Frans Hals",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Vermeer - Girl with a Red Hat.JPG",
        "title": "Girl with a Red Hat",
        "artist": "Johannes Vermeer",
        "style": "Baroque",
    },
    {
        "wikimedia_filename": "Pierre-Auguste Renoir - Girl Reading - 98.297 - Museum of Fine Arts.jpg",
        "title": "A Girl Reading",
        "artist": "Pierre-Auguste Renoir",
        "style": "Impressionism",
    },
    {
        "wikimedia_filename": "Claude Monet - Woman with a Parasol - Madame Monet and Her Son - Google Art Project.jpg",
        "title": "Woman with a Parasol",
        "artist": "Claude Monet",
        "style": "Impressionism",
    },
    {
        "wikimedia_filename": "Pierre-Auguste Renoir, The Umbrellas, ca. 1881-86.jpg",
        "title": "The Umbrellas",
        "artist": "Pierre-Auguste Renoir",
        "style": "Impressionism",
    },
    {
        "wikimedia_filename": "William-Adolphe Bouguereau (1825-1905) - The Birth of Venus (1879).jpg",
        "title": "The Birth of Venus (Bouguereau)",
        "artist": "William-Adolphe Bouguereau",
        "style": "Academic Art",
    },
    {
        "wikimedia_filename": "John William Waterhouse - The Lady of Shalott - Google Art Project.jpg",
        "title": "The Lady of Shalott",
        "artist": "John William Waterhouse",
        "style": "Pre-Raphaelite",
    },
    {
        "wikimedia_filename": "Gustav Klimt 035.jpg",
        "title": "Portrait of Adele Bloch-Bauer II",
        "artist": "Gustav Klimt",
        "style": "Art Nouveau",
    },
    {
        "wikimedia_filename": "Amedeo Modigliani Reclining Nude The Metropolitan Museum of Art.jpg",
        "title": "Reclining Nude",
        "artist": "Amedeo Modigliani",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Kirchner - Selbstbildnis als Soldat.jpg",
        "title": "Self-Portrait as a Soldier",
        "artist": "Ernst Ludwig Kirchner",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Marc Chagall, 1911, I and the Village, oil on canvas, 192.1 x 151.4 cm, Museum of Modern Art, New York.jpg",
        "title": "I and the Village",
        "artist": "Marc Chagall",
        "style": "Modernism",
    },
    {
        "wikimedia_filename": "Edvard Munch - Madonna - Google Art Project.jpg",
        "title": "Madonna",
        "artist": "Edvard Munch",
        "style": "Expressionism",
    },
    {
        "wikimedia_filename": "Edvard Munch - Vampire (1895) - Google Art Project.jpg",
        "title": "Vampire",
        "artist": "Edvard Munch",
        "style": "Expressionism",
    },
]


def sanitize_filename(artist: str, title: str) -> str:
    """Create a sanitized filename from artist and title."""
    combined = f"{artist}_{title}".lower()
    combined = re.sub(r"[^a-z0-9]+", "_", combined)
    combined = combined.strip("_")
    if len(combined) > 80:
        combined = combined[:80].rstrip("_")
    return combined


def normalize_title(title: str) -> str:
    """Normalize a title for deduplication comparison."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def get_image_url(session: requests.Session, wikimedia_filename: str) -> str | None:
    """Get a thumbnail URL from Wikimedia Commons API.

    Requests a 1024px-wide thumbnail instead of the full-resolution original
    to avoid rate-limiting and decompression bomb issues.
    """
    params = {
        "action": "query",
        "titles": f"File:{wikimedia_filename}",
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": THUMB_WIDTH,
        "format": "json",
    }
    try:
        resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                # Prefer thumbnail URL; fall back to original
                return imageinfo[0].get("thumburl") or imageinfo[0].get("url")
    except Exception as e:
        logger.warning("API error for %s: %s", wikimedia_filename, e)
    return None


def download_and_process(session: requests.Session, url: str, output_path: str) -> bool:
    """Download image from URL, resize to 512x512 RGB JPEG.

    Retries once on 429 after a longer backoff.
    """
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if attempt == 0:
                    time.sleep(5)
                    continue
                logger.warning("429 rate-limited (after retry): %s", url)
                return False
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
            img.save(output_path, "JPEG", quality=JPEG_QUALITY)
            return True
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error for %s: %s", url, e)
            return False
        except Exception as e:
            logger.warning("Download/process error for %s: %s", url, e)
            return False
    return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Read existing CSV ---
    existing_rows = []
    fieldnames = ["filename", "title", "artist", "genre", "style", "is_famous"]

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "is_famous" not in row:
                    row["is_famous"] = ""
                existing_rows.append(row)
        logger.info("Read %d existing rows from CSV.", len(existing_rows))
    else:
        logger.warning("No existing CSV found at %s", CSV_PATH)

    # --- Build deduplication index ---
    # Maps normalized title -> index in existing_rows (for updating is_famous)
    existing_title_idx: dict[str, int] = {}
    for i, row in enumerate(existing_rows):
        norm = normalize_title(row.get("title", ""))
        if norm:
            existing_title_idx[norm] = i

    # Set of all known titles (existing + newly added) to prevent any dupes
    seen_titles: set[str] = set(existing_title_idx.keys())

    # --- Set up HTTP session ---
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # --- Process paintings ---
    downloaded = 0
    skipped_dupes = 0
    failed = 0
    new_rows = []

    print(f"\nProcessing {len(FAMOUS_PAINTINGS)} famous paintings...")
    progress = tqdm(FAMOUS_PAINTINGS, desc="Famous paintings", unit="img")

    for idx, painting in enumerate(progress, start=1):
        title = painting["title"]
        artist = painting["artist"]
        style = painting["style"]
        wikimedia_filename = painting["wikimedia_filename"]

        norm_title = normalize_title(title)

        # Check for duplicates
        if norm_title in seen_titles:
            # If it exists in the original dataset, mark it as famous
            if norm_title in existing_title_idx:
                existing_rows[existing_title_idx[norm_title]]["is_famous"] = "true"
            skipped_dupes += 1
            progress.set_postfix(dl=downloaded, skip=skipped_dupes, fail=failed)
            continue

        # Get image URL from Wikimedia Commons
        image_url = get_image_url(session, wikimedia_filename)
        if not image_url:
            logger.warning("Could not get URL for: %s", title)
            failed += 1
            progress.set_postfix(dl=downloaded, skip=skipped_dupes, fail=failed)
            time.sleep(RATE_LIMIT_DELAY)
            continue

        # Build output filename
        sanitized = sanitize_filename(artist, title)
        out_filename = f"famous_{idx:04d}_{sanitized}.jpg"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

        # Download and process
        if download_and_process(session, image_url, out_path):
            downloaded += 1
            new_row = {
                "filename": out_filename,
                "title": title,
                "artist": artist,
                "genre": "portrait",
                "style": style,
                "is_famous": "true",
            }
            new_rows.append(new_row)
            seen_titles.add(norm_title)
        else:
            failed += 1

        progress.set_postfix(dl=downloaded, skip=skipped_dupes, fail=failed)
        time.sleep(RATE_LIMIT_DELAY)

    progress.close()

    # --- Write updated CSV ---
    all_rows = existing_rows + new_rows
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total in list:      {len(FAMOUS_PAINTINGS)}")
    print(f"Downloaded:         {downloaded}")
    print(f"Skipped (dupes):    {skipped_dupes}")
    print(f"Failed:             {failed}")
    print(f"New CSV total rows: {len(all_rows)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
