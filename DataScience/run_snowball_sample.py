import twitter_snowball_sampling

api = twitter_snowball_sampling.get_api()

samp = twitter_snowball_sampling.Sample(api, "Lakers", max_followers=20, max_depth=10)
samp.main()




