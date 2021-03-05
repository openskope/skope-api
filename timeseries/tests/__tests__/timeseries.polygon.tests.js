const rest = require('rest');
const mime = require('rest/interceptor/mime');

const timeseriesServiceBase = 'http://localhost:8001/timeseries-service/api/v1';

const callRESTService =  rest.wrap(mime, { mime: 'application/json' } );

describe("When a GET request selects from a region exactly covering the northwest 2x2 pixel area in the uint16 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123,45],
	    		        [-123,43],
	    		        [-121,43],
	    		        [-121,45],
	    		        [-123,45]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(105.5,3);
        expect(response.entity.values[1]).toBeCloseTo(205.5,3);
        expect(response.entity.values[2]).toBeCloseTo(305.5,3);
        expect(response.entity.values[3]).toBeCloseTo(405.5,3);
        expect(response.entity.values[4]).toBeCloseTo(505.5,3);
    });
});

describe("When a GET request selects from a region exactly covering the northwest 2x2 pixel area in the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123,45],
	    		        [-123,43],
	    		        [-121,43],
	    		        [-121,45],
	    		        [-123,45]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(105.55,3);
        expect(response.entity.values[1]).toBeCloseTo(205.55,3);
        expect(response.entity.values[2]).toBeCloseTo(305.55,3);
        expect(response.entity.values[3]).toBeCloseTo(405.55,3);
        expect(response.entity.values[4]).toBeCloseTo(505.55,3);
    });
});

describe("When a GET request selects from a region of zero area from the uint16 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });

    it ('Error type should be out of bounds', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.selectedareapolygonisnotvalid");
    });
});

describe("When a GET request selects from a region of zero area from the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	[-122.5,45.5],
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error type should be out of bounds', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.selectedareapolygonisnotvalid");
    });
});


describe("When a GET request selects from a 2x2 pixel region that intersects a 2x1 pixel region in the uint16 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
    		    	[-123,46],
    		        [-123,44],
    		        [-121,44],
    		        [-121,46],
    		        [-123,46]
	    		]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(100.5,3);
        expect(response.entity.values[1]).toBeCloseTo(200.5,3);
        expect(response.entity.values[2]).toBeCloseTo(300.5,3);
        expect(response.entity.values[3]).toBeCloseTo(400.5,3);
        expect(response.entity.values[4]).toBeCloseTo(500.5,3);
    });
});

describe("When a GET request selects from a 2x2 pixel region that intersects a 2x1 pixel region in the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
    		    	[-123,46],
    		        [-123,44],
    		        [-121,44],
    		        [-121,46],
    		        [-123,46]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(100.55,3);
        expect(response.entity.values[1]).toBeCloseTo(200.55,3);
        expect(response.entity.values[2]).toBeCloseTo(300.55,3);
        expect(response.entity.values[3]).toBeCloseTo(400.55,3);
        expect(response.entity.values[4]).toBeCloseTo(500.55,3);
    });
});


describe("When a GET request selects a region just outside coverage of the uint16 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
					"coordinates": [[
						[-124,45],
						[-124,47],
						[-123,47],
						[-123,45],
						[-124,45]
					]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error type should be out of bounds', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.selectedareaoutofbounds");
    });
});

describe("When a GET request selects a region just outside coverage of the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
    		    	[-124,45],
    		        [-124,43],
    		        [-123,43],
    		        [-123,45],
    		        [-124,45]
    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error type should be out of bounds', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.selectedareaoutofbounds");
    });
});

describe("When a GET request selects a region well outside coverage of the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
    		    	[-104,37],
    		        [-104,35],
    		        [-103,35],
    		        [-103,37],
    		        [-104,37]
    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 422', async function() {
        expect(response.status.code).toBe(422);
    });
    
    it ('Error type should be out of bounds', async function() {
        expect(response.entity.detail[0].type).toBe("value_error.selectedareaoutofbounds");
    });
});

describe("When a GET request selects exactly the top-left corner pixel of the uint16 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123.0,45],
	    		        [-123.0,44],
	    		        [-122.0,44],
	    		        [-122.0,44],
	    		        [-123.0,45]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the values of the top-left pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(100.0,3);
        expect(response.entity.values[1]).toBeCloseTo(200.0,3);
        expect(response.entity.values[2]).toBeCloseTo(300.0,3);
        expect(response.entity.values[3]).toBeCloseTo(400.0,3);
        expect(response.entity.values[4]).toBeCloseTo(500.0,3);
    });
});

describe("When a GET request selects exactly the top-left corner pixel of the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123.0,45],
	    		        [-123.0,44],
	    		        [-122.0,44],
	    		        [-122.0,45],
	    		        [-123.0,45]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the values of the top-left pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(100.0,3);
        expect(response.entity.values[1]).toBeCloseTo(200.0,3);
        expect(response.entity.values[2]).toBeCloseTo(300.0,3);
        expect(response.entity.values[3]).toBeCloseTo(400.0,3);
        expect(response.entity.values[4]).toBeCloseTo(500.0,3);
    });
});


describe("When a GET request selects 1/4 of the top-left corner pixel of the uint16 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123.0,45],
	    		        [-123.0,44.75],
	    		        [-122.75,44.75],
	    		        [-122.75,44.75],
	    		        [-123.0,45]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the values of the top-left pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(100.0,3);
        expect(response.entity.values[1]).toBeCloseTo(200.0,3);
        expect(response.entity.values[2]).toBeCloseTo(300.0,3);
        expect(response.entity.values[3]).toBeCloseTo(400.0,3);
        expect(response.entity.values[4]).toBeCloseTo(500.0,3);
    });
});

describe("When a GET request selects 1/4 of the top-left corner pixel of the float32 variable", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123.0,45],
	    		        [-123.0,44.75],
	    		        [-122.75,44.75],
	    		        [-122.75,44.75],
	    		        [-123.0,45]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the values of the top-left pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(100.0,3);
        expect(response.entity.values[1]).toBeCloseTo(200.0,3);
        expect(response.entity.values[2]).toBeCloseTo(300.0,3);
        expect(response.entity.values[3]).toBeCloseTo(400.0,3);
        expect(response.entity.values[4]).toBeCloseTo(500.0,3);
    });
});



describe("When a GET request selects from a region exactly covering the 2x2 pixel area in southwest corner the dataset", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-123,42],
	    		        [-123,40],
	    		        [-121,40],
	    		        [-121,42],
	    		        [-123,42]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(135.5,3);
        expect(response.entity.values[1]).toBeCloseTo(235.5,3);
        expect(response.entity.values[2]).toBeCloseTo(335.5,3);
        expect(response.entity.values[3]).toBeCloseTo(435.5,3);
        expect(response.entity.values[4]).toBeCloseTo(535.5,3);
    });
});

describe("When a GET request selects from a region exactly covering the 2x2 pixel area in southeast corner of uint16 variable with one NODATA pixel in each band", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-120,42],
	    		        [-118,42],
	    		        [-118,40],
	    		        [-120,40],
	    		        [-120,42]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(140.0,3);
        expect(response.entity.values[1]).toBeCloseTo(240.0,3);
        expect(response.entity.values[2]).toBeCloseTo(340.0,3);
        expect(response.entity.values[3]).toBeCloseTo(440.0,3);
        expect(response.entity.values[4]).toBeCloseTo(540.0,3);
    });
});

describe("When a GET request selects from a region exactly covering the 2x2 pixel area in southeast corner of float32 variable with one NODATA pixel in each band", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-120,42],
	    		        [-118,42],
	    		        [-118,40],
	    		        [-120,40],
	    		        [-120,42]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values).toEqual([null, null, null, null, null]);
    });
});


describe("When a GET request selects from a region exactly covering the 2x2 pixel area in uint16 variable with one NODATA pixel in 3rd band", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'uint16_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-120,43],
	    		        [-118,43],
	    		        [-118,41],
	    		        [-120,41],
	    		        [-120,43]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(126.667,3);
        expect(response.entity.values[1]).toBeCloseTo(226.667,3);
        expect(response.entity.values[2]).toBeCloseTo(328.000,3);
        expect(response.entity.values[3]).toBeCloseTo(426.667,3);
        expect(response.entity.values[4]).toBeCloseTo(526.667,3);
    });
});


describe("When a GET request selects from a region exactly covering the 2x2 pixel area in southeast corner of float32 variable with two NODATA pixels in 3rd band", async () => {
    
	var response;
	
	beforeAll(async () => {
		response = await callRESTService({
		    method: 'POST',
		    path: timeseriesServiceBase + '/timeseries',
		    entity: {
		    	datasetId: 'annual_5x5x5_dataset',
		    	variableName: 'float32_variable',
		    	boundaryGeometry: {
    		    "type": "Polygon",
    		    "coordinates": [[
	    		    	[-120,44],
	    		        [-118,44],
	    		        [-118,42],
	    		        [-120,42],
	    		        [-120,44]
	    		    	]]
	    		},
	    		start: 0,
	    		end: 4
		    }
		});
    });

    it ('HTTP response status code should be 200 - success', async function() {
        expect(response.status.code).toBe(200);
    });
    
    it ('The array should comprise the average of four pixels in each band', async function() {
        expect(response.entity.values[0]).toBeCloseTo(118.850,3);
        expect(response.entity.values[1]).toBeCloseTo(218.850,3);
        expect(response.entity.values[2]).toBe(null);
        expect(response.entity.values[3]).toBeCloseTo(418.850,3);
        expect(response.entity.values[4]).toBeCloseTo(518.850,3);
    });
});

