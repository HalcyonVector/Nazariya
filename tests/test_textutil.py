from datetime import datetime, timedelta, timezone

from nazariya.textutil import (
    canonical_url, detect_agency, detect_lang, fix_published, hamming, in_scope, section_of,
    simhash, to_signed64, to_unsigned64, tokens,
)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)


def test_canonical_url_strips_tracking_and_amp():
    assert canonical_url("http://www.example.com/news/a-story/amp/?utm_source=x&id=5#top") == \
        "https://www.example.com/news/a-story?id=5"
    assert canonical_url("https://amp.example.com/amp/news/x") == "https://example.com/news/x"
    assert canonical_url("https://timesofindia.indiatimes.com/india/x/amp_articleshow/123.cms") == \
        "https://timesofindia.indiatimes.com/india/x/articleshow/123.cms"


def test_canonical_url_keeps_id_params_and_sorts():
    assert canonical_url("https://pib.gov.in/PressReleasePage.aspx?reg=3&PRID=99") == \
        "https://pib.gov.in/PressReleasePage.aspx?PRID=99&reg=3"


def test_canonical_url_rewrites_host():
    assert canonical_url("https://thewirenews.in/rights/x", {"thewirenews.in": "thewire.in"}) == \
        "https://thewire.in/rights/x"


def test_in_scope_respects_homepage_path():
    assert in_scope("https://www.bbc.com/hindi/articles/abc", "https://www.bbc.com/hindi")
    assert not in_scope("https://www.bbc.com/bengali/articles/abc", "https://www.bbc.com/hindi")
    assert not in_scope("https://www.bbc.com/hindiish/x", "https://www.bbc.com/hindi")
    assert in_scope("https://thehindu.com/news/x", "https://www.thehindu.com/")
    assert not in_scope("https://hindi.news18.com/x", "https://www.news18.com/")


def test_section_of_skips_numbers_and_files():
    assert section_of("https://x.com/2026/10/17/india/story") == "india"
    assert section_of("https://x.com/123456.cms") is None


def test_detect_lang():
    assert detect_lang("Supreme Court hears plea on electoral rolls") == "en"
    assert detect_lang("सुप्रीम कोर्ट ने मतदाता सूची पर सुनवाई की") == "hi"
    assert detect_lang("কলকাতায় দুর্গাপুজোর মণ্ডপে") == "bn"
    # English title with a Hindi word stays English
    assert detect_lang("PM launches Swachh Bharat phase two: स्वच्छ") == "en"


def test_tokens_keep_hindi_words_whole():
    # splitting on \W would break words at vowel signs (matras)
    assert tokens("राहुल गांधी ने कहा।") == ["राहुल", "गांधी", "ने", "कहा"]


def test_simhash_near_duplicates_are_close():
    base = " ".join(f"word{i}" for i in range(300))
    edited = base.replace("word150", "changed")
    other = " ".join(f"term{i}" for i in range(300))
    assert hamming(simhash(base), simhash(edited)) <= 3
    assert hamming(simhash(base), simhash(other)) > 15


def test_signed_roundtrip():
    for x in (0, 1, (1 << 63) - 1, 1 << 63, (1 << 64) - 1):
        s = to_signed64(x)
        assert -(1 << 63) <= s < (1 << 63)
        assert to_unsigned64(s) == x


def test_detect_agency():
    assert detect_agency("New Delhi, Oct 17 (PTI) The Supreme Court on Friday...") == "PTI"
    assert detect_agency("Body text here.\n(With inputs from ANI)") == "ANI"
    assert detect_agency("नई दिल्ली, 17 अक्टूबर (भाषा) उच्चतम न्यायालय ने...") == "PTI"
    assert detect_agency("Story text.\n-- IANS") == "IANS"
    assert detect_agency("(This story has not been edited and is auto-generated from a syndicated feed.)") == "wire"
    assert detect_agency("Our correspondent reports that the minister (BJP) said...") is None


def test_fix_published():
    assert fix_published(None, NOW) == (None, False)
    ok = NOW - timedelta(hours=1)
    assert fix_published(ok, NOW) == (ok, False)
    ist_as_utc = NOW + timedelta(hours=5)  # IST wall clock mislabelled as UTC
    fixed, flag = fix_published(ist_as_utc, NOW)
    assert flag and fixed == ist_as_utc - timedelta(hours=5, minutes=30)
    far_future = NOW + timedelta(days=2)
    assert fix_published(far_future, NOW) == (NOW, True)


def test_title_from_url():
    from nazariya.textutil import title_from_url
    assert title_from_url(
        "https://timesofindia.indiatimes.com/city/patna/teenager-held-on-charge-of-raping-minor-in-west-champaran/articleshow/134969514.cms"
    ) == "Teenager held on charge of raping minor in west champaran"
    assert title_from_url("https://x.com/news/123456.cms") == ""
    assert title_from_url("https://x.com/india/pm-visits-kerala-1234567.html") == "Pm visits kerala"


def test_agency_from_author():
    from nazariya.textutil import agency_from_author
    assert agency_from_author("PTI") == "PTI"
    assert agency_from_author("Edited by Riya; ANI") == "ANI"
    assert agency_from_author("आईएएनएस") == "IANS"
    assert agency_from_author("Anirudh Sharma") is None  # "ani" inside a name is not ANI
    assert agency_from_author(None) is None


def test_live_pages_and_edit_kinds():
    from nazariya.textutil import edit_kind, is_live_page
    assert is_live_page("https://x.com/news/balen-shah-live-updates-1", "Balen Shah Live Updates: day 2")
    assert is_live_page("https://x.com/a/1", "न्यूज अपडेट्स सुबह की बड़ी खबरें")
    assert not is_live_page("https://x.com/delhi/olive-oil-prices-1", "Olive oil prices rise")
    assert edit_kind("Fire kills 3 in factory", "Fire kills 24 in factory") == "numbers changed"
    assert edit_kind("ಸ್ಟೀಕರ್ ಕಚೇರಿಗೆ ಬಂದು ರಾಜೀನಾಮೆ ಪತ್ರ ಸಲ್ಲಿಸಿದ", "ಸ್ಪೀಕರ್ ಕಚೇರಿಗೆ ಬಂದು ರಾಜೀನಾಮೆ ಪತ್ರ ಸಲ್ಲಿಸಿದ") == "small correction"
    assert edit_kind("Minister resigns over scam", "Opposition demands probe into land deal") == "rewritten"
    assert edit_kind("Youth dies in Mussoorie crash, police probe on", "Youth dies in Mussoorie crash, police probe on after family protest") == "added detail"


def test_marathi_keeps_its_language():
    from nazariya.textutil import detect_lang
    assert detect_lang("महाराष्ट्रात मुसळधार पाऊस", default="mr") == "mr"
    assert detect_lang("महाराष्ट्र में भारी बारिश", default="hi") == "hi"
    assert detect_lang("தமிழ்நாட்டில் கனமழை", default="ta") == "ta"


def test_sitemap_title_skips_image_titles():
    from lxml import etree
    from nazariya.sources import parse_sitemap_urls
    xml = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
      <url><loc>https://x.com/a</loc><image:image><image:title>epf illustration.jpg</image:title></image:image>
        <news:news><news:title>EPF hike demand grows</news:title></news:news></url>
      <url><loc>https://x.com/b</loc><image:image><image:title>p jayarajan (1).jpg</image:title></image:image></url>
    </urlset>"""
    items = parse_sitemap_urls(etree.fromstring(xml))
    assert [i.title for i in items] == ["EPF hike demand grows", ""]   # b falls back to the page title
