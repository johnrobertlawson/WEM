#!/bin/csh

set echo

set dates = (20110126 20110127 20110128 20110129 20110130 20110131)

foreach date ($dates)

	foreach hr (00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
	wget "http://nomads.ncdc.noaa.gov/data/ruc/201101/${date}/ruc2_252_${date}_${hr}00_000.grb"

end
end
