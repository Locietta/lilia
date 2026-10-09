includes("xmake/rules/*.lua")

add_requires("lighter 0.2.0", "ngcpp-proxy", "sqlite3 >=3.51.3 <4")

target("lilia-core")
    set_kind("static")
    -- public, so the generated <xmake/version/lilia.hpp> reaches dependents
    add_rules("utils.version.cpp")
    add_rules("utils.bin2obj.cpp", {extensions = {".md", ".txt"}})
    add_files("prompts/*.md")
    -- jieba's dictionary and HMM model, for learn/segment.cpp
    add_files("data/jieba/*.txt")
    -- pypinyin readings, for chat/typos.cpp
    add_files("data/pinyin/*.txt")
    add_files("config.cpp", "replay.cpp", "archive/*.cpp", "chat/*.cpp", "learn/*.cpp", "memory/*.cpp", "net/*.cpp", "onebot/*.cpp", "provider/*.cpp", "provider/detail/*.cpp", "sticker/*.cpp", "store/*.cpp", "tools/*.cpp")
    add_packages("lighter", {public = true})
    add_packages("sqlite3")
    add_packages("ngcpp-proxy", {public = true})

target("lilia")
    set_kind("binary")
    add_files("main.cpp")
    add_deps("lilia-core")

includes("tests/xmake.lua")
