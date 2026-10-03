includes("xmake/rules/*.lua")

add_requires("ngcpp-proxy", "sqlite3 >=3.51.3 <4")

target("lilia-core")
    set_kind("static")
    -- public, so the generated <xmake/version/lilia.hpp> reaches dependents
    add_rules("utils.version.cpp")
    add_files("config.cpp", "chat/*.cpp", "net/*.cpp", "onebot/*.cpp", "provider/*.cpp", "provider/detail/*.cpp", "store/*.cpp", "tools/*.cpp")
    add_deps("lighter")
    add_packages("sqlite3")
    add_packages("ngcpp-proxy", {public = true})

target("lilia")
    set_kind("binary")
    add_files("main.cpp")
    add_deps("lilia-core")

includes("tests/xmake.lua")
