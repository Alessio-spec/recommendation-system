import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:url_launcher/url_launcher.dart';

String url = 'http://127.0.0.1:5000';
// String url = 'http://10.0.2.2:5000z';
// String url = 'http://10.226.144.166:5000';

void main() => runApp(const MyApp());

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final TextEditingController _controller = TextEditingController();
  String _response = '';
  List<String> _explanations = [];
  List<String> _noRecommendations = [];
  List<String> _recommendations = [];
  String _globalUrl = '';
  String _localUrl = '';

  Map<String, String> extractUrlsFromResponse(String jsonResponse) {
    // Decoding the JSON string into a Map
    Map<String, dynamic> data = json.decode(jsonResponse);

    // Extracting URLs from the decoded Map
    String globalUrl = data['evaluations']['global'];
    String localUrl = data['evaluations']['local'];

    // Returning the URLs in a Map for easy access
    return {'global': globalUrl, 'local': localUrl};
  }

  Future<void> sendData() async {
    try {
      final response = await http.post(
        Uri.parse('$url/receive-data'),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonEncode(<String, String>{
          'data': _controller.text,
        }),
      );
      _clearState();
      if (response.statusCode == 200) {
        // _clearState();

        final Map<String, dynamic> responseData = jsonDecode(response.body);

        final urls = extractUrlsFromResponse(response.body);
        _globalUrl = urls['global'] ?? '';
        _localUrl = urls['local'] ?? '';

        if (responseData.containsKey('predictions') &&
            responseData['predictions'] is Map) {
          Map<String, dynamic> predictionsContent = responseData['predictions'];

          setState(() {
            _explanations =
                List<String>.from(predictionsContent['explanations'] ?? []);
            _noRecommendations = List<String>.from(
                predictionsContent['no_recommendation'] ?? []);
            _recommendations =
                List<String>.from(predictionsContent['recommendation'] ?? []);
          });
        } else {
          setState(() {
            _response = 'Failed to send data: ${response.statusCode}';
          });
        }
      }
    } catch (e) {
      setState(() {
        _response = 'Error: $e';
      });
    }
  }

  _clearState() {
    _controller.clear();
    setState(() {
      _explanations.clear();
      _noRecommendations.clear();
      _recommendations.clear();
    });
    _globalUrl = '';
    _localUrl = '';
  }

  Future<void> _launchURL(String urlString) async {
    if (await canLaunch(urlString)) {
      await launch(urlString);
    } else {
      throw 'Could not launch $urlString';
    }
  }

  void extractUrls(Map<String, dynamic> data) {
    // Accessing the 'evaluations' map
    Map<String, dynamic> evaluations = data['evaluations'];

    // Extracting 'global' and 'local' URLs
    String globalUrl = evaluations['global'];
    String localUrl = evaluations['local'];
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Get a recommendation'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              TextField(
                controller: _controller,
                decoration: const InputDecoration(hintText: 'Enter a date'),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 20.0),
                // Adjust the padding value as needed
                child: ElevatedButton(
                  onPressed: sendData,
                  child: const Text('Get a recommendation'),
                ),
              ),
              const SizedBox(height: 20),
              const Text('Explanations:',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              for (var item in _explanations) Text(item),
              const SizedBox(height: 20),
              const Text('No Recommendations:',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              for (var item in _noRecommendations) Text(item),
              const SizedBox(height: 20),
              const Text('Recommendations:',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              for (var item in _recommendations) Text(item),
              Padding(
                padding: const EdgeInsets.only(top: 20.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton(
                      onPressed: () => _launchURL(_globalUrl),
                      child: Text('Open Global Evaluation'),
                    ),
                    ElevatedButton(
                      onPressed: () => _launchURL(_localUrl),
                      child: Text('Open Local Evaluation'),
                    ),
                    ElevatedButton(
                      onPressed: () =>
                          _launchURL('$url/static/usage_shap_plot.html'),
                      child: Text('View plot'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
